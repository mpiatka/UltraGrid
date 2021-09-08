/**
 * @file   audio/echo.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Martin Piatka    <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2012-2021 CESNET z.s.p.o.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of CESNET nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif /* HAVE_CONFIG_H */

#include "audio/utils.h"
#include "debug.h"
#include "echo.h"

#include <speex/speex_echo.h>

#include <stdlib.h>
#include <mutex>
#include <memory>
#include "utils/ring_buffer.h"

#include "tinywav.h"

#define SAMPLES_PER_FRAME (1 << 9) //about 10ms at 48kHz, power of two for easy FFT
#define FILTER_LENGTH (48 * 500)

#define MOD_NAME "[Echo cancel] "

namespace {
        struct Ring_buf_deleter{
                void operator()(ring_buffer_t* ring) { ring_buffer_destroy(ring); }
        };

        struct Echo_state_deleter{
                void operator()(SpeexEchoState* echo) { speex_echo_state_destroy(echo); }
        };
}

struct echo_cancellation {
        std::unique_ptr<SpeexEchoState, Echo_state_deleter> echo_state;

        std::unique_ptr<ring_buffer_t, Ring_buf_deleter> near_end_ringbuf;
        std::unique_ptr<ring_buffer_t, Ring_buf_deleter> far_end_ringbuf;

        std::unique_ptr<spx_int16_t[]> frame_data;
        struct audio_frame frame;

        TinyWav tw;
        int prefill;
        bool before_first_near_sample;

        std::mutex lock;
};

static void reconfigure_echo (struct echo_cancellation *s, int sample_rate, int bps);

static void reconfigure_echo (struct echo_cancellation *s, int sample_rate, int bps)
{
        UNUSED(bps);

        s->frame.bps = 2;
        s->frame.ch_count = 1;
        s->frame.sample_rate = sample_rate;

        ring_buffer_flush(s->far_end_ringbuf.get());
        ring_buffer_flush(s->near_end_ringbuf.get());

        speex_echo_ctl(s->echo_state.get(), SPEEX_ECHO_SET_SAMPLING_RATE, &sample_rate); // should the 3rd parameter be int?

        if(tinywav_isOpen(&s->tw)){
                tinywav_close_write(&s->tw);
        }

        tinywav_open_write(&s->tw, 3, sample_rate, TW_INT16, TW_SPLIT, "uv_echo_input.wav");

}

struct echo_cancellation * echo_cancellation_init(void)
{
        struct echo_cancellation *s = new echo_cancellation();

        s->echo_state.reset(speex_echo_state_init(SAMPLES_PER_FRAME, FILTER_LENGTH));

        s->frame.data = NULL;
        s->frame.sample_rate = s->frame.bps = 0;
        std::lock_guard(s->lock);

        const int ringbuf_sample_count = 2 << 15; //should be divisable by SAMPLES_PER_FRAME
        const int bps = 2; //TODO: assuming bps to be 2

        s->far_end_ringbuf.reset(ring_buffer_init(ringbuf_sample_count * bps));
        s->near_end_ringbuf.reset(ring_buffer_init(ringbuf_sample_count * bps));

        s->frame_data = std::make_unique<spx_int16_t[]>(ringbuf_sample_count);
        s->frame.data = reinterpret_cast<char *>(s->frame_data.get());
        s->frame.max_size = ringbuf_sample_count * sizeof(s->frame_data[0]);

        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Echo cancellation initialized.\n");

        s->tw.f = NULL;

        s->prefill = 0;
        s->before_first_near_sample = true;

        return s;
}

void echo_cancellation_destroy(struct echo_cancellation *s)
{
        if(tinywav_isOpen(&s->tw)){
                tinywav_close_write(&s->tw);
        }

        delete s;
}

void echo_play(struct echo_cancellation *s, struct audio_frame *frame)
{
        std::lock_guard(s->lock);

        if(frame->ch_count != 1) {
                static int prints = 0;
                if(prints++ % 100 == 0) {
                        error_msg(MOD_NAME "Echo cancellation needs 1 played channel. Disabling echo cancellation.\n"
                                        "Use channel mapping and let only one channel played to enable this feature.\n");
                }
                return;
        }

        if(s->prefill){
                int target = (s->prefill / SAMPLES_PER_FRAME) * SAMPLES_PER_FRAME;
                int current = ring_get_current_size(s->far_end_ringbuf.get());
                //buffer can contain small remainder (<SAMPLES_PER_FRAME)
                int to_fill = target - current;
                s->prefill -= target;
                if(to_fill < 0){
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Pre fill requested to %d, but the buffer is already %d!\n", target, current);
                } else {
                        ring_advance_write_idx(s->far_end_ringbuf.get(), to_fill);
                        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Pre filling far end with %d samples\n", to_fill);
                }
        }

        size_t samples = frame->data_len / frame->bps;
        size_t ringbuf_free_samples = ring_get_available_write_size(s->far_end_ringbuf.get()) / 2;

        if(samples > ringbuf_free_samples){
                samples = ringbuf_free_samples;
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Far end ringbuf overflow!\n");
        }

        if(frame->bps != 2) {
                void *ptr1;
                int size1;
                void *ptr2;
                int size2;
                ring_get_write_regions(s->far_end_ringbuf.get(), samples * 2,
                                &ptr1, &size1, &ptr2, &size2);

                assert(size1 % 2 == 0);
                int in_bytes1 = (size1 / 2) * frame->bps;
                change_bps(static_cast<char *>(ptr1), 2, frame->data, frame->bps, in_bytes1);
                if(ptr2){
                        change_bps(static_cast<char *>(ptr2), 2, frame->data + in_bytes1, frame->bps, frame->data_len - in_bytes1);
                }

                ring_advance_write_idx(s->far_end_ringbuf.get(), samples * 2);
        } else {
                ring_buffer_write(s->far_end_ringbuf.get(), frame->data, samples * 2);
        }
}

struct audio_frame * echo_cancel(struct echo_cancellation *s, struct audio_frame *frame)
{
        struct audio_frame *res;

        std::lock_guard(s->lock);

        if(frame->ch_count != 1) {
                static int prints = 0;
                if(prints++ % 100 == 0)
                        error_msg(MOD_NAME "Echo cancellation needs 1 captured channel. Disabling echo cancellation.\n"
                                        "Use '--audio-capture-channels 1' parameter to capture single channel.\n");
                return frame;
        }


        if(frame->sample_rate != s->frame.sample_rate ||
                        frame->bps != s->frame.bps) {
                reconfigure_echo(s, frame->sample_rate, frame->bps);
        }

        size_t in_frame_samples = frame->data_len / frame->bps;

        size_t ringbuf_free_samples = ring_get_available_write_size(s->near_end_ringbuf.get()) / 2;
        if(in_frame_samples > ringbuf_free_samples){
                in_frame_samples = ringbuf_free_samples;
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Near end ringbuf overflow\n");
        }

        if(s->before_first_near_sample){
                /* It is possible that the capture thread starts late, which
                 * could create an unwanted delay between far and near ends.
                 * To partialy protect against this, drop the contents of far
                 * end buffer, when the very first near end samples arrive.
                 *
                 * This does not however protect against random capture thread
                 * freezes or dropouts.
                 */
                int current = ring_get_current_size(s->far_end_ringbuf.get());
                //drop only whole frames
                current = (current / SAMPLES_PER_FRAME) * SAMPLES_PER_FRAME;
                ring_advance_read_idx(s->far_end_ringbuf.get(), current);

                s->before_first_near_sample = false;
        }

        if(frame->bps != 2){
                //Need to change bps, put whole incoming frame into ringbuf
                void *ptr1;
                int size1;
                void *ptr2;
                int size2;
                ring_get_write_regions(s->near_end_ringbuf.get(), in_frame_samples * 2,
                                &ptr1, &size1, &ptr2, &size2);

                int in_bytes1 = (size1 / 2) * frame->bps;
                change_bps(static_cast<char *>(ptr1), 2, frame->data, frame->bps, in_bytes1);
                if(ptr2){
                        change_bps(static_cast<char *>(ptr2), 2, frame->data + in_bytes1, frame->bps, frame->data_len - in_bytes1);
                }
                ring_advance_write_idx(s->near_end_ringbuf.get(), in_frame_samples * 2);
        } else {
                ring_buffer_write(s->near_end_ringbuf.get(), frame->data, frame->data_len);
        }

        size_t near_end_samples = ring_get_current_size(s->near_end_ringbuf.get()) / 2;
        size_t far_end_samples = ring_get_current_size(s->far_end_ringbuf.get()) / 2;

        if(far_end_samples < near_end_samples){
                log_msg(LOG_LEVEL_INFO, MOD_NAME "Not enough far end samples (%lu near, %lu far)\n", near_end_samples, far_end_samples);

                //The delay between far end and near end will always be at least
                //recorded frame length
                s->prefill = in_frame_samples;
        }

        size_t frames_to_process = near_end_samples / SAMPLES_PER_FRAME;
        if(!frames_to_process){
                return NULL;
        }

        size_t out_size = frames_to_process * SAMPLES_PER_FRAME * 2;
        assert(s->frame.max_size >= out_size);
        s->frame.data_len = out_size;
        res = &s->frame;

        spx_int16_t *out_ptr = (spx_int16_t *)(void *) s->frame.data;
        for(size_t i = 0; i < frames_to_process; i++){
                spx_int16_t near_arr[SAMPLES_PER_FRAME];
                spx_int16_t far_arr[SAMPLES_PER_FRAME];

                if(far_end_samples >= SAMPLES_PER_FRAME){
                        ring_buffer_read(s->far_end_ringbuf.get(), reinterpret_cast<char *>(far_arr), SAMPLES_PER_FRAME * 2);
                        ring_buffer_read(s->near_end_ringbuf.get(), reinterpret_cast<char *>(near_arr), SAMPLES_PER_FRAME * 2);

                        speex_echo_cancellation(s->echo_state.get(), near_arr, far_arr, out_ptr); 
                        far_end_samples -= SAMPLES_PER_FRAME;
                } else {
                        ring_buffer_read(s->near_end_ringbuf.get(), reinterpret_cast<char *>(out_ptr), SAMPLES_PER_FRAME * 2);
                }

                float left_channel[SAMPLES_PER_FRAME];
                float right_channel[SAMPLES_PER_FRAME];
                float result_channel[SAMPLES_PER_FRAME];

                for(int i = 0; i < SAMPLES_PER_FRAME; i++){
                        left_channel[i] = near_arr[i] / 32767.0f;
                        right_channel[i] = far_arr[i] / 32767.0f;
                        result_channel[i] = out_ptr[i] / 32767.0f;
                }

                void *channels[] = {left_channel, right_channel, result_channel};

                tinywav_write_f(&s->tw, channels, SAMPLES_PER_FRAME);

                out_ptr += SAMPLES_PER_FRAME;
        }

        return res;
}

