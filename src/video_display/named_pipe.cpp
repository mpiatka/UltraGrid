/**
 * @file   video_display/named_pipe.cpp
 * @author Martin Piatka    <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2022 CESNET, z. s. p. o.
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

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "lib_common.h"
#include "video.h"
#include "video_display.h"
#include "video_codec.h"

#include <fstream>
#include <condition_variable>
#include <chrono>
#include <list>
#include <map>
#include <vector>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <cmath>

static constexpr unsigned int IN_QUEUE_MAX_BUFFER_LEN = 5;
static constexpr int SKIP_FIRST_N_FRAMES_IN_STREAM = 5;

namespace{
struct frame_deleter { void operator()(video_frame *f){ vf_free(f); } };
using unique_frame = std::unique_ptr<video_frame, frame_deleter>;
}

struct state_named_pipe {
        std::queue<unique_frame> incoming_queue;
        std::condition_variable frame_consumed_cv;
        std::condition_variable frame_available_cv;
        std::mutex lock;

        struct video_desc desc;
        struct video_desc display_desc;

        std::ofstream out;

        struct module *parent;
};

static void show_help(){
        printf("named pipe display\n");
}

static void *display_named_pipe_init(struct module *parent, const char *fmt, unsigned int flags)
{
        UNUSED(flags);
        UNUSED(fmt);

        auto s = std::make_unique<state_named_pipe>();

        s->parent = parent;

        s->out.open("/tmp/fifo", std::ios::binary);

        if(!s->out.good()){
                log_msg(LOG_LEVEL_FATAL, "Failed to open fifo\n");
                return nullptr;
        }

        return s.release();
}

static void write_frame(state_named_pipe *s, video_frame *f){
        std::array<char, 128> header;
        header.fill(0);

        *reinterpret_cast<uint32_t *>(&header[0]) = f->tiles[0].width;
        *reinterpret_cast<uint32_t *>(&header[4]) = f->tiles[0].height;
        *reinterpret_cast<uint32_t *>(&header[8]) = f->tiles[0].data_len;
        *reinterpret_cast<uint32_t *>(&header[12]) = f->color_spec;

        s->out.write(header.data(), header.size());
        s->out.write(f->tiles[0].data, f->tiles[0].data_len);
}

static void display_named_pipe_run(void *state)
{
        auto s = static_cast<state_named_pipe *>(state);
        int skipped = 0;

        while (1) {
                auto frame = [&]{
                        std::unique_lock<std::mutex> l(s->lock);
                        s->frame_available_cv.wait(l,
                                        [s]{return s->incoming_queue.size() > 0;});
                        auto frame = std::move(s->incoming_queue.front());
                        s->incoming_queue.pop();
                        s->frame_consumed_cv.notify_one();
                        return frame;
                }();

                if (!frame) {
                        break;
                }

                if (skipped < SKIP_FIRST_N_FRAMES_IN_STREAM){
                        skipped++;
                        continue;
                }

                write_frame(s, frame.get());

        }
}

static void display_named_pipe_done(void *state)
{
        auto s = static_cast<state_named_pipe *>(state);
        delete s;
}

static struct video_frame *display_named_pipe_getf(void *state)
{
        auto s = static_cast<state_named_pipe *>(state);

        return vf_alloc_desc_data(s->desc);
}

static int display_named_pipe_putf(void *state, struct video_frame *frame, int flags)
{
        auto s = static_cast<state_named_pipe *>(state);
        auto f = unique_frame(frame);

        if (flags == PUTF_DISCARD)
                return 0;

        std::unique_lock<std::mutex> lg(s->lock);
        if (s->incoming_queue.size() >= IN_QUEUE_MAX_BUFFER_LEN)
                log_msg(LOG_LEVEL_WARNING, "Named pipe: queue full!\n");

        if (flags == PUTF_NONBLOCK && s->incoming_queue.size() >= IN_QUEUE_MAX_BUFFER_LEN)
                return 1;

        s->frame_consumed_cv.wait(lg, [s]{return s->incoming_queue.size() < IN_QUEUE_MAX_BUFFER_LEN;});
        s->incoming_queue.push(std::move(f));
        lg.unlock();
        s->frame_available_cv.notify_one();

        return 0;
}

static int display_named_pipe_get_property(void *state, int property, void *val, size_t *len)
{
        UNUSED(state);
        codec_t codecs[] = {UYVY, RGBA, RGB};
        enum interlacing_t supported_il_modes[] = {PROGRESSIVE, INTERLACED_MERGED, SEGMENTED_FRAME};
        int rgb_shift[] = {0, 8, 16};

        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        if(sizeof(codecs) <= *len) {
                                memcpy(val, codecs, sizeof(codecs));
                        } else {
                                return FALSE;
                        }

                        *len = sizeof(codecs);
                        break;
                case DISPLAY_PROPERTY_RGB_SHIFT:
                        if(sizeof(rgb_shift) > *len) {
                                return FALSE;
                        }
                        memcpy(val, rgb_shift, sizeof(rgb_shift));
                        *len = sizeof(rgb_shift);
                        break;
                case DISPLAY_PROPERTY_BUF_PITCH:
                        *(int *) val = PITCH_DEFAULT;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_SUPPORTED_IL_MODES:
                        if(sizeof(supported_il_modes) <= *len) {
                                memcpy(val, supported_il_modes, sizeof(supported_il_modes));
                        } else {
                                return FALSE;
                        }
                        *len = sizeof(supported_il_modes);
                        break;
                default:
                        return FALSE;
        }
        return TRUE;
}

static int display_named_pipe_reconfigure(void *state, struct video_desc desc)
{
        auto s = static_cast<state_named_pipe *>(state);

        s->desc = desc;

        return 1;
}

static void display_named_pipe_put_audio_frame(void *state, struct audio_frame *frame)
{
        UNUSED(state);
        UNUSED(frame);
}

static int display_named_pipe_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate)
{
        UNUSED(state);
        UNUSED(quant_samples);
        UNUSED(channels);
        UNUSED(sample_rate);

        return FALSE;
}

static const struct video_display_info display_named_pipe_info = {
        [](struct device_info **available_cards, int *count, void (**deleter)(void *)) {
                UNUSED(deleter);
                *available_cards = nullptr;
                *count = 0;
        },
        display_named_pipe_init,
        display_named_pipe_run,
        display_named_pipe_done,
        display_named_pipe_getf,
        display_named_pipe_putf,
        display_named_pipe_reconfigure,
        display_named_pipe_get_property,
        display_named_pipe_put_audio_frame,
        display_named_pipe_reconfigure_audio,
        DISPLAY_DOESNT_NEED_MAINLOOP,
};

REGISTER_HIDDEN_MODULE(named_pipe, &display_named_pipe_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);

