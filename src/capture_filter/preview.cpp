/**
 * @file   capture_filter/preview.cpp
 * @author Martin Piatka    <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2018 CESNET, z. s. p. o.
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

#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>

#include "capture_filter.h"

#include "debug.h"
#include "lib_common.h"
#include "utils/color_out.h"
#include "utils/misc.h"
#include "utils/sv_parse_num.hpp"
#include "video.h"
#include "video_codec.h"
#include "tools/ipc_frame.h"
#include "tools/ipc_frame_unix.h"
#include "tools/ipc_frame_ug.h"

struct module;

static int init(struct module *parent, const char *cfg, void **state);
static void done(void *state);
static struct video_frame *filter(void *state, struct video_frame *in);

struct state_preview_filter{
        std::mutex mut;
        std::condition_variable frame_submitted_cv;

        std::vector<Ipc_frame_uniq> free_frames;
        std::queue<Ipc_frame_uniq> frame_queue;

        int target_width = 960;
        int target_height = 540;

        std::thread worker_thread;
};

static void worker(struct state_preview_filter *s, std::string path){
        Ipc_frame_uniq frame;
        Ipc_frame_writer_uniq writer;

        for(;;){
                if(!writer){
                        writer.reset(ipc_frame_writer_new(path.c_str()));
                        if(!writer){
                                sleep(1);
                                continue;
                        }
                }

                {
                        std::unique_lock<std::mutex> lock(s->mut);
                        s->frame_submitted_cv.wait(lock,
                                        [=]{ return !s->frame_queue.empty(); });
                        frame = std::move(s->frame_queue.front());
                        s->frame_queue.pop();
                }

                if(!frame)
                        break;

                if(!ipc_frame_writer_write(writer.get(), frame.get())){;
                        writer.reset();
                }

                std::lock_guard<std::mutex> lock(s->mut);
                s->free_frames.push_back(std::move(frame));
        }
}


static int init(struct module *parent, const char *cfg, void **state){
        UNUSED(parent);
        if (strlen(cfg) > 0) {
                std::cout << RED(BOLD("preview")) << " capture filter serves as a proxy for passing frames to GUI\n";
                return strcmp(cfg, "help") == 0 ? 1 : -1;
        }

        struct state_preview_filter *s = new state_preview_filter();

        s->free_frames.emplace_back(ipc_frame_new());
        s->free_frames.emplace_back(ipc_frame_new());

        std::string_view cfg_sv = cfg;

        std::string socket_path = "/tmp/ug_preview_cap_unix";

        std::string_view tok;
        tok = tokenize(cfg_sv, ':');
        if(!tok.empty()) socket_path = tok;
        tok = tokenize(cfg_sv, ':');
        parse_num(tok, s->target_width);
        parse_num(tok, s->target_height);

        s->worker_thread = std::thread(worker, s, socket_path);

        *state = s;

        return 0;
}

static void done(void *state){
        auto s = static_cast<state_preview_filter *> (state);

        {
                std::lock_guard<std::mutex> lock(s->mut);
                s->frame_queue.push(nullptr);
        }
        s->frame_submitted_cv.notify_one();
        s->worker_thread.join();

        delete s;
}

static struct video_frame *filter(void *state, struct video_frame *in){
        struct state_preview_filter *s = (state_preview_filter *) state;

        Ipc_frame_uniq ipc_frame;
        {
                std::lock_guard<std::mutex> lock(s->mut);
                if(!s->free_frames.empty()){
                        ipc_frame = std::move(s->free_frames.back());
                        s->free_frames.pop_back();
                }
        }

        if(!ipc_frame)
                return in;

        assert(in->tile_count == 1);
        const tile *tile = &in->tiles[0];

        float scale = 0;
        if(s->target_width != -1 && s->target_height != -1){
                scale = (static_cast<float>(tile->width) * tile->height)
                        / (static_cast<float>(s->target_width) * s->target_height);

                if(scale < 1)
                        scale = 1;
                scale = std::round(scale);
        }

        log_msg(LOG_LEVEL_NOTICE, "Scale %f\n", scale);


        if(ipc_frame_from_ug_frame(ipc_frame.get(), in, RGB, (int) scale)){
                std::lock_guard<std::mutex> lock(s->mut);
                s->frame_queue.push(std::move(ipc_frame));
                s->frame_submitted_cv.notify_one();
        } else {
                log_msg(LOG_LEVEL_WARNING, "Unable to convert\n");
        }

        return in;
}


static const struct capture_filter_info capture_filter_preview = {
        .init = init,
        .done = done,
        .filter = filter,
};

REGISTER_HIDDEN_MODULE(preview, &capture_filter_preview, LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);

