/**
 * @file   video_display/preview.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Martin Piatka    <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2014-2021 CESNET, z. s. p. o.
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
#include "video_codec.h"
#include "tools/ipc_frame.h"
#include "tools/ipc_frame_ug.h"

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
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>

using namespace std;

static constexpr unsigned int IN_QUEUE_MAX_BUFFER_LEN = 5;
static constexpr int SKIP_FIRST_N_FRAMES_IN_STREAM = 5;

struct state_preview_display_common {
        ~state_preview_display_common() {
                if(out_sock > 0)
                        close(out_sock);
        }

        queue<struct video_frame *> incoming_queue;
        condition_variable in_queue_decremented_cv;

        mutex lock;
        condition_variable cv;

        Ipc_frame_uniq ipc_frame;
        int out_sock = -1;

        int scaledW, scaledH;
        int scaleF;

        struct module *parent;
};

struct state_preview_display {
        shared_ptr<struct state_preview_display_common> common;
        struct video_desc desc;
};

static void show_help(){
        printf("Preview display\n");
        printf("Internal use by GUI only\n");
}

static void *display_preview_init(struct module *parent, const char *fmt, unsigned int flags)
{
        UNUSED(flags);
        struct state_preview_display *s;

        s = new state_preview_display();

        if (fmt && strlen(fmt) > 0) {
                if (isdigit(fmt[0])) { // fork
                        struct state_preview_display *orig;
                        sscanf(fmt, "%p", &orig);
                        s->common = orig->common;
                        return s;
                } else {
                        show_help();
                        delete s;
                        return &display_init_noerr;
                }
        }
        s->common = shared_ptr<state_preview_display_common>(new state_preview_display_common());
        s->common->parent = parent;

        sockaddr_un addr;
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        snprintf(addr.sun_path, sizeof(addr.sun_path), "/tmp/ug_preview_disp_unix");

        int data_socket = socket(AF_UNIX, SOCK_STREAM, 0);

        int ret = connect(data_socket, (const struct sockaddr *) &addr, sizeof(addr));
        if(ret == -1){
                log_msg(LOG_LEVEL_FATAL, "Unable to connect to preview socket\n");
                exit(EXIT_FAILURE);
        }

        s->common->out_sock = data_socket;
        s->common->ipc_frame.reset(ipc_frame_new());
        return s;
}

static void display_preview_run(void *state)
{
        shared_ptr<struct state_preview_display_common> s = ((struct state_preview_display *)state)->common;
        int skipped = 0;

        while (1) {
                struct video_frame *frame;
                {
                        unique_lock<mutex> lg(s->lock);
                        s->cv.wait(lg, [s]{return s->incoming_queue.size() > 0;});
                        frame = s->incoming_queue.front();
                        s->incoming_queue.pop();
                        s->in_queue_decremented_cv.notify_one();
                }

                if (!frame) {
                        break;
                }

                if (skipped < SKIP_FIRST_N_FRAMES_IN_STREAM){
                        skipped++;
                        vf_free(frame);
                        continue;
                }

                assert(frame->tile_count == 1);
                const tile *tile = &frame->tiles[0];

                const float target_width = 960;
                const float target_height = 540;
                float scale = ((tile->width / target_width) + (tile->height / target_height)) / 2.f;
                if(scale < 1)
                        scale = 1;
                scale = std::round(scale);

                if(!ipc_frame_from_ug_frame(s->ipc_frame.get(), frame, RGB, (int) scale)){
                        log_msg(LOG_LEVEL_WARNING, "Unable to convert\n");
                        continue;
                }

                errno = 0;
                if(!ipc_frame_write_to_fd(s->ipc_frame.get(), s->out_sock)){
                        perror("Unable to send frame");
                        exit(1);
                }

                vf_free(frame);
        }
}

static void display_preview_done(void *state)
{
        struct state_preview_display *s = (struct state_preview_display *)state;
        delete s;
}

static struct video_frame *display_preview_getf(void *state)
{
        struct state_preview_display *s = (struct state_preview_display *)state;

        return vf_alloc_desc_data(s->desc);
}

static int display_preview_putf(void *state, struct video_frame *frame, int flags)
{
        shared_ptr<struct state_preview_display_common> s = ((struct state_preview_display *)state)->common;

        if (flags == PUTF_DISCARD) {
                vf_free(frame);
        } else {
                unique_lock<mutex> lg(s->lock);
                if (s->incoming_queue.size() >= IN_QUEUE_MAX_BUFFER_LEN) {
                        fprintf(stderr, "Preview: queue full!\n");
                }
                if (flags == PUTF_NONBLOCK && s->incoming_queue.size() >= IN_QUEUE_MAX_BUFFER_LEN) {
                        vf_free(frame);
                        return 1;
                }
                s->in_queue_decremented_cv.wait(lg, [s]{return s->incoming_queue.size() < IN_QUEUE_MAX_BUFFER_LEN;});
                s->incoming_queue.push(frame);
                lg.unlock();
                s->cv.notify_one();
        }

        return 0;
}

static int display_preview_get_property(void *state, int property, void *val, size_t *len)
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

static int display_preview_reconfigure(void *state, struct video_desc desc)
{
        struct state_preview_display *s = (struct state_preview_display *) state;

        s->desc = desc;

        return 1;
}

static void display_preview_put_audio_frame(void *state, const struct audio_frame *frame)
{
        UNUSED(state);
        UNUSED(frame);
}

static int display_preview_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate)
{
        UNUSED(state);
        UNUSED(quant_samples);
        UNUSED(channels);
        UNUSED(sample_rate);

        return FALSE;
}

static const struct video_display_info display_preview_info = {
        [](struct device_info **available_cards, int *count, void (**deleter)(void *)) {
                UNUSED(deleter);
                *available_cards = nullptr;
                *count = 0;
        },
        display_preview_init,
        display_preview_run,
        display_preview_done,
        display_preview_getf,
        display_preview_putf,
        display_preview_reconfigure,
        display_preview_get_property,
        display_preview_put_audio_frame,
        display_preview_reconfigure_audio,
        DISPLAY_DOESNT_NEED_MAINLOOP,
};

REGISTER_HIDDEN_MODULE(preview, &display_preview_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);

