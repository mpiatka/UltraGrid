/**
 * @file   video_capture/omt.cpp
 * @author Martin Piatka     <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2026 CESNET, zájmové sdružení právnických osob
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

#include <algorithm>
#include <cassert>
#include <memory>
#include <thread>

#include "debug.h"
#include "omt_common.hpp"
#include "video_capture.h"
#include "lib_common.h"
#include "video_codec.h"
#include "video_frame.h"
#include "audio/types.h"
#include "audio/utils.h"
#include "utils/color_out.h"
#include "utils/string_view_utils.hpp"
#include "utils/thread.h"

#define MOD_NAME "[omt] "

namespace{
using video_frame_uniq = std::unique_ptr<video_frame, deleter_from_fcn<vf_free>>;

struct Audio_frame_data{
        audio_frame f{};
        std::unique_ptr<char[]> data;
};

template<typename T>
class Double_buffer{
public:
        T& front() noexcept{ return bufs[idx]; }
        T& back() noexcept{ return bufs[idx ^ 1]; }

        void swap() noexcept{ idx ^= 1; }
private:
        T bufs[2];
        uint8_t idx = 0;
};

struct state_omt_cap{
        ~state_omt_cap(){
                should_exit = true;
                if(audio_recv_thread.joinable()){
                        audio_recv_thread.join();
                }
        }

        omt_receive_uniq omt_h;

        video_frame_uniq ug_frame;

        std::mutex mutex;
        Double_buffer<Audio_frame_data> audio_f;

        std::atomic<bool> should_exit = false;
        std::thread audio_recv_thread;

        std::string sender_addr = "localhost";
        int port = 6400;
};

void capture_omt_probe(device_info **available_cards, int *count, void(** deleter)(void *)){
        *deleter = free;
        *available_cards = nullptr;
        *count = 0;
}

void show_help(){
        color_printf("Open Media Transport\n");
        color_printf("Usage\n");
        color_printf(TERM_BOLD TERM_FG_RED "\t-t omt" TERM_FG_RESET "[:addr=<addr>][:port=<port>]\n" TERM_RESET);
        color_printf("\n");
}

int parse_cfg(state_omt_cap& s, std::string_view cfg){
        while(!cfg.empty()){
                auto tok = tokenize(cfg, ':', '"');
                const auto key = tokenize(tok, '=', '"');
                const auto val = tokenize(tok, '=', '"');

                if(key == "addr"){
                        s.sender_addr = val;
                } else if(key == "port"){
                        if(!parse_num(val, s.port)){
                                log_msg(LOG_LEVEL_FATAL, "Failed to parse port: %s\n", std::string(val).c_str());
                                return VIDCAP_INIT_FAIL;
                        }
                } else if(key == "help"){
                        show_help();
                        return VIDCAP_INIT_NOERR;
                } else{
                        log_msg(LOG_LEVEL_FATAL, "Unknown parameter: %s\n", std::string(key).c_str());
                        return VIDCAP_INIT_FAIL;
                }
        }

        return VIDCAP_INIT_OK;
}

void init_audio_frame(Audio_frame_data& f, const OMTMediaFrame& omt){
        assert(omt.Type == OMTFrameType_Audio);

        f.f.ch_count = omt.Channels;
        f.f.bps = 2;
        f.f.sample_rate = omt.SampleRate;

        f.f.max_size = f.f.ch_count * f.f.bps * f.f.sample_rate;
        f.data.reset(new char[f.f.max_size]);
        f.f.data = f.data.get();
        f.f.data_len = 0;
}

void float2S16(short *out , const float *in, int samples) {
        for(int i = 0; i < samples; i++) {
                float sample = in[i];
                if(sample < -1.f) sample = -1.f;
                if(sample > 1.f) sample = 1.f;
                out[i] = INT16_MAX * sample;
        }
}

void audio_frame_append(Audio_frame_data& f, const OMTMediaFrame& omt_audio){
        constexpr int S16_BPS = 2;
        const auto ch_count = omt_audio.Channels;

        auto dst = reinterpret_cast<int16_t *>(f.f.data + f.f.data_len);
        auto src = static_cast<const float *>(omt_audio.Data);
        auto samples_left = omt_audio.SamplesPerChannel;
        const auto frame_capacity = (f.f.max_size - f.f.data_len) / f.f.bps / ch_count;
        if(samples_left > frame_capacity){
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Buffer overflow, dropping %d samples\n", samples_left - frame_capacity);
                samples_left = frame_capacity;
        }
        f.f.data_len += samples_left * f.f.bps * ch_count;
        while(samples_left > 0){
                constexpr auto chunk_samples = 128;
                const auto to_process = std::min(chunk_samples, samples_left);

                for(int ch = 0; ch < ch_count; ch++){
                        alignas(32) int16_t S16samples[chunk_samples];
                        const float *ch_src = src + ch * omt_audio.SamplesPerChannel;

                        float2S16(S16samples, ch_src, to_process);

                        mux_channel(reinterpret_cast<char *>(dst),
                                reinterpret_cast<const char *>(S16samples),
                                S16_BPS, to_process * S16_BPS, ch_count, ch, 1.0);
                }

                dst += to_process * ch_count;
                src += to_process;
                samples_left -= to_process;
        }
}

void omt_cap_audio_worker(state_omt_cap *s){
        set_thread_name(__func__);

        while(!s->should_exit){
                const auto omt_audio = omt_receive(s->omt_h.get(), OMTFrameType_Audio, 10);
                if(!omt_audio)
                        continue;

                std::scoped_lock lk(s->mutex);
                auto& f = s->audio_f.back();

                if(!f.data
                        || f.f.ch_count != omt_audio->Channels
                        || f.f.sample_rate != omt_audio->SampleRate)
                {
                        init_audio_frame(f, *omt_audio);
                }

                audio_frame_append(f, *omt_audio);
        }
}

int capture_omt_init(const vidcap_params *params, void **state){
        auto s = std::make_unique<state_omt_cap>();

        ug_register_omt_log_callback();

        if(auto parse_ret = parse_cfg(*s, vidcap_params_get_fmt(params)); parse_ret != VIDCAP_INIT_OK){
                return parse_ret;
        }

        std::string addr = "omt://" + s->sender_addr + ":" + std::to_string(s->port);

        s->omt_h.reset(omt_receive_create(addr.c_str(), static_cast<OMTFrameType>(OMTFrameType_Audio | OMTFrameType_Video),
                OMTPreferredVideoFormat_UYVY, OMTReceiveFlags_None));

        s->audio_recv_thread = std::thread(omt_cap_audio_worker, s.get());

        *state = s.release();
        return VIDCAP_INIT_OK;
}

void capture_omt_done(void *state){
        auto s = static_cast<state_omt_cap *>(state);
        delete s;
}

video_frame *capture_omt_grab(void *state, audio_frame **audio){
        auto s = static_cast<state_omt_cap *>(state);

        {
                std::scoped_lock lk(s->mutex);
                s->audio_f.front().f.data_len = 0;
                s->audio_f.swap();

                if(auto& audio_frame = s->audio_f.front(); audio_frame.f.data_len != 0){
                        *audio = &audio_frame.f;
                }
        }

        const auto omt_frame = omt_receive(s->omt_h.get(), OMTFrameType_Video, 100);
        if(!omt_frame){
                return nullptr;
        }

        if(const video_desc incoming_desc = video_desc_from_omt_frame(omt_frame);
                !s->ug_frame || video_desc_eq(incoming_desc, video_desc_from_frame(s->ug_frame.get())))
        {
                s->ug_frame.reset(vf_alloc_desc(incoming_desc));
        }

        assert(omt_frame->Stride == vc_get_linesize(s->ug_frame->tiles[0].width, s->ug_frame->color_spec));
        s->ug_frame->tiles[0].data = static_cast<char *>(omt_frame->Data);
        s->ug_frame->tiles[0].data_len = omt_frame->DataLength;

        return s->ug_frame.get();
}

constexpr video_capture_info vidcap_omt_info = {
        capture_omt_probe,
        capture_omt_init,
        capture_omt_done,
        capture_omt_grab,
        MOD_NAME,
};
}

REGISTER_MODULE(omt, &vidcap_omt_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);
