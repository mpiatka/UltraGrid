/**
 * @file   video_display/unix_sock.cpp
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

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>

static constexpr unsigned int IN_QUEUE_MAX_BUFFER_LEN = 5;
static constexpr int SKIP_FIRST_N_FRAMES_IN_STREAM = 5;

namespace{
struct frame_deleter { void operator()(video_frame *f){ vf_free(f); } };
using unique_frame = std::unique_ptr<video_frame, frame_deleter>;
}

struct state_unix_sock {
        std::queue<unique_frame> incoming_queue;
        std::condition_variable frame_consumed_cv;
        std::condition_variable frame_available_cv;
        std::mutex lock;

        struct video_desc desc;
        struct video_desc display_desc;

        int out_sock;

        struct module *parent;
};

static void show_help(){
        printf("unix socket display\n");
}

static void *display_unix_sock_init(struct module *parent, const char *fmt, unsigned int flags)
{
        UNUSED(flags);
        UNUSED(fmt);

        auto s = std::make_unique<state_unix_sock>();

        sockaddr_un addr;
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        snprintf(addr.sun_path, sizeof(addr.sun_path), "/tmp/ug_unix");

        int data_socket = socket(AF_UNIX, SOCK_STREAM, 0);

        int ret = connect(data_socket, (const struct sockaddr *) &addr, sizeof(addr));
        if(ret == -1){
                exit(EXIT_FAILURE);
        }

        s->parent = parent;
        s->out_sock = data_socket;


        return s.release();
}

static void block_write(int fd, void *buf, size_t size){
        size_t written = 0;
        char *src = static_cast<char *>(buf);

        while(written < size){
                int ret = send(fd, src + written, size - written, MSG_NOSIGNAL);
                if(ret == -1)
                        return;
                written += ret;
        }
}

static void write_int(char *dst, uint32_t val){
        auto src = reinterpret_cast<char *>(&val);
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = src[3];
}

static void write_frame(state_unix_sock *s, video_frame *f){
        std::array<char, 128> header;
        header.fill(0);

        write_int(header.data() + 0, f->tiles[0].width);
        write_int(header.data() + 4, f->tiles[0].height);
        write_int(header.data() + 8, f->tiles[0].data_len);
        write_int(header.data() + 12, f->color_spec);

        errno = 0;
        block_write(s->out_sock, header.data(), header.size());
        block_write(s->out_sock, f->tiles[0].data, f->tiles[0].data_len);
        if(errno == EPIPE){
                perror("Disconnect");
                exit(1);
        }
}

static void display_unix_sock_run(void *state)
{
        auto s = static_cast<state_unix_sock *>(state);
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

static void display_unix_sock_done(void *state)
{
        auto s = static_cast<state_unix_sock *>(state);
        delete s;
}

static struct video_frame *display_unix_sock_getf(void *state)
{
        auto s = static_cast<state_unix_sock *>(state);

        return vf_alloc_desc_data(s->desc);
}

static int display_unix_sock_putf(void *state, struct video_frame *frame, int flags)
{
        auto s = static_cast<state_unix_sock *>(state);
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

static int display_unix_sock_get_property(void *state, int property, void *val, size_t *len)
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

static int display_unix_sock_reconfigure(void *state, struct video_desc desc)
{
        auto s = static_cast<state_unix_sock *>(state);

        s->desc = desc;

        return 1;
}

static void display_unix_sock_put_audio_frame(void *state, struct audio_frame *frame)
{
        UNUSED(state);
        UNUSED(frame);
}

static int display_unix_sock_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate)
{
        UNUSED(state);
        UNUSED(quant_samples);
        UNUSED(channels);
        UNUSED(sample_rate);

        return FALSE;
}

static const struct video_display_info display_unix_sock_info = {
        [](struct device_info **available_cards, int *count, void (**deleter)(void *)) {
                UNUSED(deleter);
                *available_cards = nullptr;
                *count = 0;
        },
        display_unix_sock_init,
        display_unix_sock_run,
        display_unix_sock_done,
        display_unix_sock_getf,
        display_unix_sock_putf,
        display_unix_sock_reconfigure,
        display_unix_sock_get_property,
        display_unix_sock_put_audio_frame,
        display_unix_sock_reconfigure_audio,
        DISPLAY_DOESNT_NEED_MAINLOOP,
};

REGISTER_HIDDEN_MODULE(unix_sock, &display_unix_sock_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);

