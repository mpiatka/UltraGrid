/**
 * @file   video_display/rpi4_out.cpp
 * @author Martin Piatka    <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2021 CESNET, z. s. p. o.
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
#include "rang.hpp"
#include "video.h"
#include "video_codec.h"
#include "video_display.h"
#include "hwaccel_rpi4.h"

#include <memory>
#include <queue>
#include <stack>
#include <mutex>
#include <condition_variable>
#include <stdexcept>

#include <bcm_host.h>
#include <interface/mmal/mmal.h>
#include <interface/mmal/mmal_component.h>
#include <interface/mmal/util/mmal_default_components.h>

#define MAX_BUFFER_SIZE 3

namespace{

struct frame_deleter{
        void operator()(struct video_frame *f){
                vf_free(f);
        }
};

struct mmal_component_deleter{
        void operator()(MMAL_COMPONENT_T *c){
                mmal_component_destroy(c);
        }
};

using mmal_component_unique = std::unique_ptr<MMAL_COMPONENT_T, mmal_component_deleter>;

struct mmal_pool_deleter{
        void operator()(MMAL_POOL_T *p){
                mmal_pool_destroy(p);
        }
};

using mmal_pool_unique = std::unique_ptr<MMAL_POOL_T, mmal_pool_deleter>;

class Rpi4_video_out{
public:
        Rpi4_video_out() = default;
        Rpi4_video_out(int x, int y, int width, int height, bool fs, int layer);
private:
        void set_output_params();

        int out_pos_x;
        int out_pos_y;
        int out_width;
        int out_height;
        bool fullscreen;
        int layer;

        mmal_component_unique renderer_component;
        mmal_pool_unique pool;
};

void Rpi4_video_out::set_output_params(){
        MMAL_DISPLAYREGION_T region = {};
        region.hdr = {MMAL_PARAMETER_DISPLAYREGION, sizeof(region)};
        region.set = MMAL_DISPLAY_SET_DEST_RECT
                | MMAL_DISPLAY_SET_FULLSCREEN
                | MMAL_DISPLAY_SET_LAYER
                | MMAL_DISPLAY_SET_ALPHA;
        region.dest_rect = {out_pos_x, out_pos_y, out_width, out_height};
        region.fullscreen = fullscreen;
        region.layer = layer;
        region.alpha = 0xff;

        mmal_port_parameter_set(renderer_component->input[0], &region.hdr);
}

Rpi4_video_out::Rpi4_video_out(int x, int y, int width, int height, bool fs, int layer):
        out_pos_x(x),
        out_pos_y(y),
        out_width(width),
        out_height(height),
        fullscreen(fs),
        layer(layer)
{
        bcm_host_init();

        MMAL_COMPONENT_T *c = nullptr;
        if(mmal_component_create(MMAL_COMPONENT_DEFAULT_VIDEO_RENDERER, &c) != MMAL_SUCCESS){
                throw std::runtime_error("Failed to create renderer component");
        }
        renderer_component.reset(c);

        set_output_params();

        if(mmal_component_enable(renderer_component.get()) != MMAL_SUCCESS){
                throw std::runtime_error("Failed to enable renderer component");
        }

}

} //anonymous namespace

using unique_frame = std::unique_ptr<struct video_frame, frame_deleter>;

struct rpi4_display_state{
        struct video_desc current_desc;

        std::mutex frame_queue_mut;
        std::condition_variable new_frame_ready_cv;
        std::condition_variable frame_consumed_cv;
        std::queue<unique_frame> frame_queue;

        std::mutex free_frames_mut;
        std::stack<unique_frame> free_frames;

        Rpi4_video_out video_out;
};

static void *display_rpi4_init(struct module *parent, const char *cfg, unsigned int flags)
{
        rpi4_display_state *s = new rpi4_display_state();
        s->video_out = Rpi4_video_out(0, 0, 1280, 720, false, 2);
        return s;
}

static void display_rpi4_done(void *state) {
        auto *s = static_cast<rpi4_display_state *>(state);

        delete s;
}

static void frame_data_deleter(struct video_frame *buf){
        auto wrapper = reinterpret_cast<av_frame_wrapper *>(buf->tiles[0].data);

        av_frame_free(&wrapper->av_frame);

        log_msg(LOG_LEVEL_NOTICE, "RPI frame delete\n");

        delete wrapper;
}

static struct video_frame *alloc_new_frame(rpi4_display_state *s){
        auto new_frame = vf_alloc_desc(s->current_desc);

        assert(new_frame->tile_count == 1);

        auto wrapper = new av_frame_wrapper();

        wrapper->av_frame = av_frame_alloc();

        new_frame->tiles[0].data_len = sizeof(av_frame_wrapper);
        new_frame->tiles[0].data = reinterpret_cast<char *>(wrapper);
        new_frame->callbacks.recycle = av_frame_wrapper_recycle; 
        new_frame->callbacks.copy = av_frame_wrapper_copy; 
        new_frame->callbacks.data_deleter = frame_data_deleter; 

        return new_frame;
}

static struct video_frame *display_rpi4_getf(void *state) {
        auto *s = static_cast<rpi4_display_state *>(state);

        {//lock
                std::lock_guard lk(s->free_frames_mut);

                while(!s->free_frames.empty()){
                        unique_frame frame = std::move(s->free_frames.top());
                        s->free_frames.pop();
                        if(video_desc_eq(video_desc_from_frame(frame.get()), s->current_desc))
                        {
                                return frame.release();
                        }
                }
        }

        auto new_frame = alloc_new_frame(s);
        return new_frame;
}

static int display_rpi4_putf(void *state, struct video_frame *frame, int flags)
{
        auto *s = static_cast<rpi4_display_state *>(state);

        if(!frame){
                std::unique_lock lk(s->frame_queue_mut);
                s->frame_queue.emplace(frame);
                lk.unlock();
                s->new_frame_ready_cv.notify_one();
                return 0;
        }

        if (flags == PUTF_DISCARD) {
                vf_recycle(frame);
                std::lock_guard(s->free_frames_mut);
                s->free_frames.emplace(frame);
                return 0;
        }

        if (s->frame_queue.size() >= MAX_BUFFER_SIZE && flags == PUTF_NONBLOCK) {
                log_msg(LOG_LEVEL_VERBOSE, "nonblock putf drop\n");
                vf_recycle(frame);
                std::lock_guard(s->free_frames_mut);
                s->free_frames.emplace(frame);
                return 1;
        }

        std::unique_lock lk(s->frame_queue_mut);
        s->frame_consumed_cv.wait(lk, [s]{return s->frame_queue.size() < MAX_BUFFER_SIZE;});
        s->frame_queue.emplace(frame);
        lk.unlock();
        s->new_frame_ready_cv.notify_one();

        return 0;
}

static void display_rpi4_run(void *state)
{
        auto *s = static_cast<rpi4_display_state *>(state);

        log_msg(LOG_LEVEL_NOTICE, "RPI4 out run\n");

        bool run = true;

        while(run){
                std::unique_lock lk(s->frame_queue_mut);
                s->new_frame_ready_cv.wait(lk, [s] {return s->frame_queue.size() > 0;});

                if (s->frame_queue.size() == 0) {
                        continue;
                }

                unique_frame frame = std::move(s->frame_queue.front());
                s->frame_queue.pop();
                lk.unlock();
                s->frame_consumed_cv.notify_one();

                if(!frame){
                        run = false;
                        break;
                }

                auto av_wrap = reinterpret_cast<struct av_frame_wrapper *>(frame->tiles[0].data);

                vf_recycle(frame.get());
                std::lock_guard(s->free_frames_mut);
                s->free_frames.push(std::move(frame));
        }
}

static int display_rpi4_reconfigure(void *state, struct video_desc desc)
{
        auto *s = static_cast<rpi4_display_state *>(state);

        assert(desc.color_spec == RPI4_8);
        s->current_desc = desc;

        return TRUE;
}

static auto display_rpi4_get_property(void *state, int property, void *val, size_t *len)
{
        UNUSED(state);
        codec_t codecs[] = {
                RPI4_8,
        };
        enum interlacing_t supported_il_modes[] = {PROGRESSIVE};
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

static void display_rpi4_put_audio_frame(void *, struct audio_frame *)
{
}

static int display_rpi4_reconfigure_audio(void *, int, int, int)
{
        return false;
}

static const struct video_display_info display_rpi4_info = {
        [](struct device_info **available_cards, int *count, void (**deleter)(void *)) {
                UNUSED(deleter);
                *available_cards = nullptr;
                *count = 0;
        },
        display_rpi4_init,
        display_rpi4_run,
        display_rpi4_done,
        display_rpi4_getf,
        display_rpi4_putf,
        display_rpi4_reconfigure,
        display_rpi4_get_property,
        display_rpi4_put_audio_frame,
        display_rpi4_reconfigure_audio,
        DISPLAY_DOESNT_NEED_MAINLOOP,
};

REGISTER_MODULE(rpi4, &display_rpi4_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);
