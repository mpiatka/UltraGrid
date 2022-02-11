/**
 * @file   video_display/conference.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Martin Piatka    <445597@mail.muni.cz>
 */
/*
 * Copyright (c) 2014-2022 CESNET, z. s. p. o.
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

#include <cinttypes>
#include <condition_variable>
#include <chrono>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <thread>

#include "utils/profile_timer.hpp"

#define MOD_NAME "[conference] "

namespace{

struct frame_deleter{ void operator()(video_frame *f){ vf_free(f); } };

}//anon namespace

static constexpr std::chrono::milliseconds SOURCE_TIMEOUT(500);
static constexpr unsigned int IN_QUEUE_MAX_BUFFER_LEN = 5;

struct state_conference_common{
        state_conference_common() = default;
        ~state_conference_common(){

        };


        struct module *parent = nullptr;

        int width;
        int height;
        double fps;
        struct video_desc display_desc = {};

        std::thread real_display_thread;
        struct display *real_display = {};

        std::mutex incoming_frames_lock;
        std::condition_variable incoming_frame_consumed;
        std::condition_variable new_incoming_frame_cv;
        std::queue<std::unique_ptr<video_frame, frame_deleter>> incoming_frames;
};

struct state_conference {
        std::shared_ptr<state_conference_common> common;
        struct video_desc desc;
};

static struct display *display_conference_fork(void *state)
{
        auto s = ((struct state_conference *)state)->common;
        struct display *out;
        char fmt[2 + sizeof(void *) * 2 + 1] = "";
        snprintf(fmt, sizeof fmt, "%p", state);

        int rc = initialize_video_display(s->parent,
                        "conference", fmt, 0, NULL, &out);
        if (rc == 0) return out; else return NULL;

        return out;
}

static void show_help(){
        printf("Conference display\n");
        printf("Usage:\n");
        printf("\t-d conference:<display_config>#<width>:<height>:[fps]\n");
}

static void real_display_runner(struct display *disp) {
        display_run(disp);
}

static void *display_conference_init(struct module *parent, const char *fmt, unsigned int flags)
{
        auto s = std::make_unique<state_conference>();
        char *fmt_copy = NULL;
        const char *requested_display = "gl";
        const char *cfg = NULL;

        log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "init fmt: %s\n", fmt);

        int width;
        int height;
        double fps;

        if(fmt && strlen(fmt) > 0){
                if (isdigit(fmt[0])) { // fork
                        struct state_conference *orig;
                        sscanf(fmt, "%p", &orig);
                        s->common = orig->common;
                        return s.release();
                } else {
                        char *tmp = strdup(fmt);
                        char *save_ptr = NULL;
                        char *item;

                        item = strtok_r(tmp, "#", &save_ptr);
                        if(!item || strlen(item) == 0){
                                show_help();
                                free(tmp);
                                return &display_init_noerr;
                        }
                        //Display configuration
                        fmt_copy = strdup(item);
                        requested_display = fmt_copy;
                        char *delim = strchr(fmt_copy, ':');
                        if (delim) {
                                *delim = '\0';
                                cfg = delim + 1;
                        }
                        item = strtok_r(NULL, "#", &save_ptr);
                        //Conference configuration
                        if(!item || strlen(item) == 0){
                                show_help();
                                free(fmt_copy);
                                free(tmp);
                                return &display_init_noerr;
                        }
                        width = atoi(item);
                        item = strchr(item, ':');
                        if(!item || strlen(item + 1) == 0){
                                show_help();
                                free(fmt_copy);
                                free(tmp);
                                return &display_init_noerr;
                        }
                        height = atoi(++item);
                        if((item = strchr(item, ':'))){
                                fps = atoi(++item);
                        }
                        free(tmp);
                }
        } else {
                show_help();
                return &display_init_noerr;
        }

        s->common = std::make_shared<state_conference_common>();
        s->common->parent = parent;

        int ret = initialize_video_display(parent, requested_display, cfg,
                        flags, nullptr, &s->common->real_display);

        if(ret != 0){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to init real display\n");
                return nullptr;
        }
        s->common->real_display_thread = std::thread(real_display_runner,
                        s->common->real_display);

        return s.release();
}

static void check_reconf(struct state_conference_common *s, struct video_desc desc)
{
        if (!video_desc_eq(desc, s->display_desc)) {
                s->display_desc = desc;
                fprintf(stderr, "RECONFIGURED\n");
                display_reconfigure(s->real_display, s->display_desc, VIDEO_NORMAL);
        }
}

static video_desc get_video_desc(std::shared_ptr<struct state_conference_common> s){
        video_desc desc;
        desc.width = s->width;
        desc.height = s->height;
        desc.fps = s->fps;
        desc.color_spec = UYVY;
        desc.interlacing = PROGRESSIVE;
        desc.tile_count = 1;

        return desc;
}

static auto extract_incoming_frame(state_conference_common *s){
        std::unique_lock<std::mutex> lock(s->incoming_frames_lock);
        s->new_incoming_frame_cv.wait(lock,
                        [s]{ return !s->incoming_frames.empty(); });
        auto frame = std::move(s->incoming_frames.front());
        s->incoming_frames.pop();
        lock.unlock();
        s->incoming_frame_consumed.notify_one();

        return frame;
}

static void display_conference_run(void *state)
{
        PROFILE_FUNC;
        auto s = static_cast<state_conference *>(state)->common;

        for(;;){
                auto frame = extract_incoming_frame(s.get());

                if(!frame){
                        display_put_frame(s->real_display, nullptr, PUTF_BLOCKING);
                        break;
                }

                auto now = std::chrono::steady_clock::now();
        }

        return;
}

static int display_conference_get_property(void *state, int property, void *val, size_t *len)
{
        auto s = ((struct state_conference *)state)->common;
        if (property == DISPLAY_PROPERTY_SUPPORTS_MULTI_SOURCES) {
                ((struct multi_sources_supp_info *) val)->val = true;
                ((struct multi_sources_supp_info *) val)->fork_display = display_conference_fork;
                ((struct multi_sources_supp_info *) val)->state = state;
                *len = sizeof(struct multi_sources_supp_info);
                return TRUE;

        } else {
                return display_ctl_property(s->real_display, property, val, len);
        }
}

static int display_conference_reconfigure(void *state, struct video_desc desc)
{
        struct state_conference *s = (struct state_conference *) state;

        s->desc = desc;

        return 1;
}

static void display_conference_put_audio_frame(void *state, struct audio_frame *frame)
{
        UNUSED(state);
        UNUSED(frame);
}

static int display_conference_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate)
{
        UNUSED(state);
        UNUSED(quant_samples);
        UNUSED(channels);
        UNUSED(sample_rate);

        return FALSE;
}

static void display_conference_done(void *state)
{
        auto s = static_cast<state_conference *>(state);
        delete s;
}

static struct video_frame *display_conference_getf(void *state)
{
        auto s = static_cast<state_conference *>(state);

        return vf_alloc_desc_data(s->desc);
}

static int display_conference_putf(void *state, struct video_frame *frame, int flags)
{
        auto s = static_cast<state_conference *>(state)->common;

        if (flags == PUTF_DISCARD) {
                vf_free(frame);
        } else {
                std::unique_lock<std::mutex> lg(s->incoming_frames_lock);
                if (s->incoming_frames.size() >= IN_QUEUE_MAX_BUFFER_LEN) {
                        log_msg(LOG_LEVEL_WARNING, "[conference] queue full!\n");
                        if(flags == PUTF_NONBLOCK){
                                vf_free(frame);
                                return 1;
                        }
                }
                s->incoming_frame_consumed.wait(lg,
                                [s]{
                                return s->incoming_frames.size() < IN_QUEUE_MAX_BUFFER_LEN;
                                });
                s->incoming_frames.emplace(frame);
                lg.unlock();    
                s->new_incoming_frame_cv.notify_one();
        }

        return 0;
}

static const struct video_display_info display_conference_info = {
        [](struct device_info **available_cards, int *count, void (**deleter)(void *)) {
                UNUSED(deleter);
                *available_cards = nullptr;
                *count = 0;
        },
        display_conference_init,
        display_conference_run,
        display_conference_done,
        display_conference_getf,
        display_conference_putf,
        display_conference_reconfigure,
        display_conference_get_property,
        display_conference_put_audio_frame,
        display_conference_reconfigure_audio,
        DISPLAY_DOESNT_NEED_MAINLOOP,
};

REGISTER_HIDDEN_MODULE(conference, &display_conference_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);

