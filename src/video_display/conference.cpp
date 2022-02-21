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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"
#pragma GCC diagnostic ignored "-Wcast-qual"
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#pragma GCC diagnostic pop

#ifdef __SSSE3__
#include "tmmintrin.h"
// compat with older Clang compiler
#ifndef _mm_bslli_si128
#define _mm_bslli_si128 _mm_slli_si128
#endif
#ifndef _mm_bsrli_si128
#define _mm_bsrli_si128 _mm_srli_si128
#endif
#endif

#include "utils/profile_timer.hpp"

#define MOD_NAME "[conference] "

namespace{
using clock = std::chrono::steady_clock;

struct frame_deleter{ void operator()(video_frame *f){ vf_free(f); } };
using unique_frame = std::unique_ptr<video_frame, frame_deleter>;

struct Participant{
        void to_cv_frame();
        unique_frame frame;
        clock::time_point last_time_recieved;

        int x = 0;
        int y = 0;
        int width = 0;
        int height = 0;

        cv::Mat luma;
        cv::Mat chroma;
};

void Participant::to_cv_frame(){
        if(!frame)
                return;

        assert(frame->color_spec == UYVY);
        assert(frame->tile_count == 1);

        PROFILE_FUNC;

        auto& frame_tile = frame->tiles[0];
        luma.create(cv::Size(frame_tile.width, frame_tile.height), CV_8UC1);
        chroma.create(cv::Size(frame_tile.width / 2, frame_tile.height), CV_8UC2);

        assert(luma.isContinuous() && chroma.isContinuous());
        unsigned char *src = reinterpret_cast<unsigned char *>(frame_tile.data);
        unsigned char *luma_dst = luma.ptr(0);
        unsigned char *chroma_dst = chroma.ptr(0);

        unsigned src_len = frame_tile.data_len;

#ifdef __SSSE3__
        __m128i uv_shuff = _mm_set_epi8(255, 255, 255, 255, 255, 255, 255, 255, 14, 12, 10, 8, 6, 4, 2, 0);
        __m128i y_shuff = _mm_set_epi8(255, 255, 255, 255, 255, 255, 255, 255, 15, 13, 11, 9, 7, 5, 3, 1);
        while(src_len >= 32){
                __m128i uyvy = _mm_lddqu_si128((__m128i *)(void *) src);
                src += 16;

                __m128i y_low = _mm_shuffle_epi8(uyvy, y_shuff);
                __m128i uv_low = _mm_shuffle_epi8(uyvy, uv_shuff);
                
                uyvy = _mm_lddqu_si128((__m128i *)(void *) src);
                src += 16;

                __m128i y_high = _mm_shuffle_epi8(uyvy, y_shuff);
                __m128i uv_high = _mm_shuffle_epi8(uyvy, uv_shuff);

                _mm_store_si128((__m128i *)(void *) luma_dst, _mm_or_si128(y_low, _mm_slli_si128(y_high, 8)));
                luma_dst += 16;
                _mm_store_si128((__m128i *)(void *) chroma_dst, _mm_or_si128(uv_low, _mm_slli_si128(uv_high, 8)));
                chroma_dst += 16;

                src_len -= 32;
        }
#endif

        while(src_len >= 4){
                *chroma_dst++ = *src++;
                *luma_dst++ = *src++;
                *chroma_dst++ = *src++;
                *luma_dst++ = *src++;

                src_len -= 4;
        }
}

class Video_mixer{
public:
        Video_mixer(int width, int height, codec_t color_space);

        void process_frame(unique_frame&& f);
        void get_mixed(video_frame *result);

private:
        void compute_layout();
        int width;
        int height;
        codec_t color_space;

        cv::Mat mixed_luma;
        cv::Mat mixed_chroma;

        std::map<uint32_t, Participant> participants;
};

Video_mixer::Video_mixer(int width, int height, codec_t color_space):
        width(width),
        height(height),
        color_space(color_space)
{
        mixed_luma.create(cv::Size(width, height), CV_8UC1);
        mixed_chroma.create(cv::Size(width / 2, height), CV_8UC2);
}

void Video_mixer::compute_layout(){
        unsigned tileW;
        unsigned tileH;

        if(participants.size() == 1){
                auto& t = (*participants.begin()).second;
                t.x = 0;
                t.y = 0;
                t.width = width;
                t.height = height;
                return;
        }

        unsigned rows = (unsigned) ceil(sqrt(participants.size()));
        tileW = width / rows;
        tileH = height / rows;

        unsigned i = 0;
        int pos = 0;
        for(auto& [ssrc, t]: participants){
                t.x = (pos % rows) * tileW;
                t.y = (pos / rows) * tileH;

                t.width = tileW;
                t.height = tileH;

                i++;
                pos++;
        }
}

void Video_mixer::process_frame(unique_frame&& f){
        auto iter = participants.find(f->ssrc);
        auto& p = participants[f->ssrc];
        p.frame = std::move(f);
        p.last_time_recieved = clock::now();

        if(iter == participants.end()){
                compute_layout();
                mixed_luma.setTo(16);
                mixed_chroma.setTo(128);
        }
}

void Video_mixer::get_mixed(video_frame *result){
        PROFILE_FUNC;

        auto now = clock::now();
        for(auto it = participants.begin(); it != participants.end();){
                auto& p = it->second;
                if(now - p.last_time_recieved > std::chrono::seconds(2)){
                        it = participants.erase(it);
                        compute_layout();
                        continue;
                }

                p.to_cv_frame();

                PROFILE_DETAIL("resize participant");
                cv::Size l_size(p.width, p.height);
                cv::Size c_size(p.width / 2, p.height);
                cv::resize(p.luma, mixed_luma(cv::Rect(p.x, p.y, p.width, p.height)), l_size, 0, 0);
                cv::resize(p.chroma, mixed_chroma(cv::Rect(p.x / 2, p.y, p.width / 2, p.height)), c_size, 0, 0);
                ++it;
                PROFILE_DETAIL("");
        }

        unsigned char *dst = reinterpret_cast<unsigned char *>(result->tiles[0].data);
        unsigned char *chroma_src = mixed_chroma.ptr(0);
        unsigned char *luma_src = mixed_luma.ptr(0);
        assert(mixed_luma.isContinuous() && mixed_chroma.isContinuous());
        PROFILE_DETAIL("Convert to ug frame");
        unsigned dst_len = result->tiles[0].data_len;

#ifdef __SSSE3__
        __m128i y_shuff = _mm_set_epi8(7, 0xFF, 6, 0xFF, 5, 0xFF, 4, 0xFF, 3, 0xFF, 2, 0xFF, 1, 0xFF, 0, 0xFF);
        __m128i uv_shuff = _mm_set_epi8(0xFF, 7, 0xFF, 6, 0xFF, 5, 0xFF, 4, 0xFF, 3, 0xFF, 2, 0xFF, 1, 0xFF, 0);
        while(dst_len >= 32){
               __m128i luma = _mm_load_si128((__m128i const*)(const void *) luma_src); 
               luma_src += 16;
               __m128i chroma = _mm_load_si128((__m128i const*)(const void *) chroma_src); 
               chroma_src += 16;

               __m128i res = _mm_or_si128(_mm_shuffle_epi8(luma, y_shuff), _mm_shuffle_epi8(chroma, uv_shuff));
               _mm_storeu_si128((__m128i *)(void *) dst, res);
               dst += 16;

               luma = _mm_srli_si128(luma, 8);
               chroma = _mm_srli_si128(chroma, 8);

               res = _mm_or_si128(_mm_shuffle_epi8(luma, y_shuff), _mm_shuffle_epi8(chroma, uv_shuff));
               _mm_storeu_si128((__m128i *)(void *) dst, res);
               dst += 16;

               dst_len -= 32;
        }
#endif

        while(dst_len >= 4){
                *dst++ = *chroma_src++;
                *dst++ = *luma_src++;
                *dst++ = *chroma_src++;
                *dst++ = *luma_src++;

                dst_len -= 4;
        }
}

struct display_deleter{ void operator()(display *d){ display_done(d); } };
using unique_disp = std::unique_ptr<display, display_deleter>;
}//anon namespace

static constexpr std::chrono::milliseconds SOURCE_TIMEOUT(500);
static constexpr unsigned int IN_QUEUE_MAX_BUFFER_LEN = 5;

struct state_conference_common{
        struct module *parent = nullptr;

        struct video_desc desc = {};

        unique_disp real_display;
        struct video_desc display_desc = {};

        std::mutex incoming_frames_lock;
        std::condition_variable incoming_frame_consumed;
        std::condition_variable new_incoming_frame_cv;
        std::queue<unique_frame> incoming_frames;
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

static void *display_conference_init(struct module *parent, const char *fmt, unsigned int flags)
{
        auto s = std::make_unique<state_conference>();
        char *fmt_copy = NULL;
        const char *requested_display = "gl";
        const char *cfg = "";

        log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "init fmt: %s\n", fmt);

        video_desc desc;
        desc.color_spec = UYVY;
        desc.interlacing = PROGRESSIVE;
        desc.tile_count = 1;
        desc.fps = 0;

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
                        desc.width = atoi(item);
                        item = strchr(item, ':');
                        if(!item || strlen(item + 1) == 0){
                                show_help();
                                free(fmt_copy);
                                free(tmp);
                                return &display_init_noerr;
                        }
                        desc.height = atoi(++item);
                        if((item = strchr(item, ':'))){
                                desc.fps = atoi(++item);
                        }
                        free(tmp);
                }
        } else {
                show_help();
                return &display_init_noerr;
        }

        s->common = std::make_shared<state_conference_common>();
        s->common->parent = parent;
        s->common->desc = desc;

        struct display *d_ptr;
        int ret = initialize_video_display(parent, requested_display, cfg,
                        flags, nullptr, &d_ptr);

        if(ret != 0){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to init real display\n");
                return nullptr;
        }

        s->common->real_display.reset(d_ptr);

        return s.release();
}

static void check_reconf(struct state_conference_common *s, struct video_desc desc)
{
        if (!video_desc_eq(desc, s->display_desc)) {
                s->display_desc = desc;
                log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "reconfiguring real display\n");
                display_reconfigure(s->real_display.get(), s->display_desc, VIDEO_NORMAL);
        }
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

static void display_conference_worker(std::shared_ptr<state_conference_common> s){
        PROFILE_FUNC;
        Video_mixer mixer(s->desc.width, s->desc.height, UYVY);

        auto next_frame_time = clock::now();
        for(;;){
                auto frame = extract_incoming_frame(s.get());
                if(!frame){
                        display_put_frame(s->real_display.get(), nullptr, PUTF_BLOCKING);
                        break;
                }

                mixer.process_frame(std::move(frame));

                auto now = clock::now();
                if(next_frame_time <= now){
                        check_reconf(s.get(), s->desc);
                        auto disp_frame = display_get_frame(s->real_display.get());

                        mixer.get_mixed(disp_frame);

                        display_put_frame(s->real_display.get(), disp_frame, PUTF_BLOCKING);

                        using namespace std::chrono_literals;
                        next_frame_time += std::chrono::duration_cast<clock::duration>(1s / s->desc.fps);
                        if(next_frame_time < now){
                                //log_msg(LOG_LEVEL_WARNING, MOD_NAME "unable to keep up");
                                next_frame_time = now;
                        }
                }
        }

        return;
}

static void display_conference_run(void *state)
{
        PROFILE_FUNC;
        auto s = static_cast<state_conference *>(state)->common;

        std::thread worker = std::thread(display_conference_worker, s);

        display_run_this_thread(s->real_display.get());

        worker.join();
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

        } else if(property == DISPLAY_PROPERTY_CODECS) {
                codec_t codecs[] = {UYVY};

                memcpy(val, codecs, sizeof(codecs));

                *len = sizeof(codecs);

                return TRUE;
        }
        
        return display_ctl_property(s->real_display.get(), property, val, len);
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
        PROFILE_FUNC;
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

static auto display_conference_needs_mainloop(void *state)
{
        auto s = static_cast<struct state_conference *>(state)->common;
        return display_needs_mainloop(s->real_display.get());
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
        display_conference_needs_mainloop
};

REGISTER_HIDDEN_MODULE(conference, &display_conference_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);

