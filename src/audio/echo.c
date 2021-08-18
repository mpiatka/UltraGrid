/**
 * @file   audio/echo.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2012-2017 CESNET z.s.p.o.
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
#include <pthread.h>
#include "pa_ringbuffer.h"

#define SAMPLES_PER_FRAME (48 * 10)
#define FILTER_LENGTH (48 * 500)

struct echo_cancellation {
        SpeexEchoState *echo_state;

        PaUtilRingBuffer near_end_ringbuf;
        char *near_end_ringbuf_data;
        PaUtilRingBuffer far_end_ringbuf;
        char *far_end_ringbuf_data;

        struct audio_frame frame;

        pthread_mutex_t lock;
};

static void reconfigure_echo (struct echo_cancellation *s, int sample_rate, int bps);

static void reconfigure_echo (struct echo_cancellation *s, int sample_rate, int bps)
{
        UNUSED(bps);

        s->frame.bps = 2;
        s->frame.ch_count = 1;
        s->frame.sample_rate = sample_rate;
        s->frame.max_size = s->frame.data_len = 0;
        free(s->frame.data);
        s->frame.data = NULL;


        PaUtil_FlushRingBuffer(&s->far_end_ringbuf);
        PaUtil_FlushRingBuffer(&s->near_end_ringbuf);

        speex_echo_ctl(s->echo_state, SPEEX_ECHO_SET_SAMPLING_RATE, &sample_rate); // should the 3rd parameter be int?
}

struct echo_cancellation * echo_cancellation_init(void)
{
        struct echo_cancellation *s = (struct echo_cancellation *) calloc(1, sizeof(struct echo_cancellation));

        s->echo_state = speex_echo_state_init(SAMPLES_PER_FRAME, FILTER_LENGTH);

        s->frame.data = NULL;
        s->frame.sample_rate = s->frame.bps = 0;
        pthread_mutex_init(&s->lock, NULL);

        const int ringbuf_sample_count = 2 << 15;
        const int bps = 2; //TODO: assuming bps to be 2
        s->far_end_ringbuf_data = malloc(ringbuf_sample_count * bps);
        s->near_end_ringbuf_data = malloc(ringbuf_sample_count * bps);

        PaUtil_InitializeRingBuffer(&s->far_end_ringbuf, bps, ringbuf_sample_count, s->far_end_ringbuf_data);
        PaUtil_InitializeRingBuffer(&s->near_end_ringbuf, bps, ringbuf_sample_count, s->near_end_ringbuf_data);

        printf("Echo cancellation initialized.\n");

        return s;
}

void echo_cancellation_destroy(struct echo_cancellation *s)
{
        if(s->echo_state) {
                speex_echo_state_destroy(s->echo_state);  
        }
        free(s->near_end_ringbuf_data);
        free(s->far_end_ringbuf_data);

        pthread_mutex_destroy(&s->lock);

        free(s);
}

void echo_play(struct echo_cancellation *s, struct audio_frame *frame)
{
        pthread_mutex_lock(&s->lock);

        if(frame->ch_count != 1) {
                static int prints = 0;
                if(prints++ % 100 == 0) {
                        fprintf(stderr, "Echo cancellation needs 1 played channel. Disabling echo cancellation.\n"
                                        "Use channel mapping and let only one channel played to enable this feature.\n");
                }
                pthread_mutex_unlock(&s->lock);
                return;
        }

        size_t samples = frame->data_len / frame->bps;
        size_t written = 0;
        if(frame->bps != 2) {
                char *tmp = malloc(samples * 2);
                change_bps(tmp, 2, frame->data, frame->bps, frame->data_len/* bytes */);

                written = PaUtil_WriteRingBuffer(&s->far_end_ringbuf, tmp, samples);

                free(tmp);
        } else {
                printf("Play request, saving %d\n", frame->data_len);

                written = PaUtil_WriteRingBuffer(&s->far_end_ringbuf, frame->data, samples);
        }

        if(written != samples){
                printf("Far end ringbuf overflow!\n");
        }

        pthread_mutex_unlock(&s->lock);
}

struct audio_frame * echo_cancel(struct echo_cancellation *s, struct audio_frame *frame)
{
        struct audio_frame *res;

        pthread_mutex_lock(&s->lock);

        if(frame->ch_count != 1) {
                static int prints = 0;
                if(prints++ % 100 == 0)
                        fprintf(stderr, "Echo cancellation needs 1 captured channel. Disabling echo cancellation.\n"
                                        "Use '--audio-capture-channels 1' parameter to capture single channel.\n");
                pthread_mutex_unlock(&s->lock);
                return frame;
        }


        if(frame->sample_rate != s->frame.sample_rate ||
                        frame->bps != s->frame.bps) {
                reconfigure_echo(s, frame->sample_rate, frame->bps);
        }

        char *data;
        char *tmp;
        int data_len;

        if(frame->bps != 2) {
                data_len = frame->data_len / frame->bps * 2;
                data = tmp = malloc(data_len);
                change_bps(tmp, 2, frame->data, frame->bps, frame->data_len/* bytes */);
        } else {
                tmp = NULL;
                data = frame->data;
                data_len = frame->data_len;
        }

        size_t samples = data_len / 2;
        size_t written = PaUtil_WriteRingBuffer(&s->near_end_ringbuf, data, samples);

        free(tmp);

        if(written != samples){
                printf("Near end ringbuf overflow\n");
        }

        size_t near_end_samples = PaUtil_GetRingBufferReadAvailable(&s->near_end_ringbuf);
        size_t far_end_samples = PaUtil_GetRingBufferReadAvailable(&s->far_end_ringbuf);
        size_t available_samples = (near_end_samples > far_end_samples) ? far_end_samples : near_end_samples;

        //free(s->frame.data);

        size_t frames_to_process = available_samples / SAMPLES_PER_FRAME;
        if(!frames_to_process){
                pthread_mutex_unlock(&s->lock);
                return NULL;
        }

        s->frame.data = malloc(available_samples * 2);
        s->frame.max_size = available_samples * 2;
        s->frame.data_len = available_samples * 2;

        res = &s->frame;

        spx_int16_t *out_ptr = (spx_int16_t *)(void *) s->frame.data;
        for(size_t i = 0; i < frames_to_process; i++){
                spx_int16_t near_arr[SAMPLES_PER_FRAME];
                spx_int16_t far_arr[SAMPLES_PER_FRAME];

                PaUtil_ReadRingBuffer(&s->near_end_ringbuf, near_arr, SAMPLES_PER_FRAME);
                PaUtil_ReadRingBuffer(&s->far_end_ringbuf, far_arr, SAMPLES_PER_FRAME);
                available_samples -= SAMPLES_PER_FRAME;

                speex_echo_cancellation(s->echo_state, near_arr, far_arr, out_ptr); 
        }

        pthread_mutex_unlock(&s->lock);

        return res;
}

