/**
 * @file   remap.cpp
 * @author Martin Piatka     <piatka@cesnet.cz>
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif /* HAVE_CONFIG_H */

#include <memory>
#include <vector>
#include <string_view>
#include <charconv>
#include <stdio.h>

#include "debug.h"
#include "audio/audio_filter.h"
#include "audio/types.h"
#include "audio/utils.h"
#include "lib_common.h"
#include "utils/misc.h"

struct state_remap{
        int ch_count;

        std::vector<char> data;
        struct audio_frame buffer;

        std::vector<std::vector<int>> in_to_out_ch_map;
        std::vector<int> out_ch_contributors;
};

static void usage(){
        printf("Remaps audio channels:\n\n");
        printf("remap usage:\n");
        printf("\tremap:<src_ch>:<dst_ch>[,<src_ch>:<dst_ch>]...\n\n");
        printf("Example:\n");
        printf("swap stereo: remap:0:1,1:0\n");
        printf("mix to mono: remap:0:0,1:0\n");
}

template<typename T>
static bool parse_num(std::string_view str, T& num){
        return std::from_chars(str.begin(), str.end(), num).ec == std::errc();
}

static af_result_code init(const char *cfg, void **state){
        state_remap *s = new state_remap();
        *state = s;

        std::string_view conf(cfg);

        if(conf.empty()){
                log_msg(LOG_LEVEL_ERROR, "No mapping configured!\n");
                usage();
                return AF_FAILURE;
        }

        while(!conf.empty()){
                std::string_view tok = tokenize(conf, ',');
                if(tok == "help"){
                        usage();
                        return AF_HELP_SHOWN;
                }

                auto src_t = tokenize(tok, ':');
                auto dst_t = tokenize(tok, ':');
                if(src_t.empty() || dst_t.empty())
                        return AF_FAILURE;

                unsigned src_idx;
                unsigned dst_idx;
                if(!parse_num(src_t, src_idx) || !parse_num(dst_t, dst_idx)){
                        return AF_FAILURE;
                }

                if(src_idx >= s->in_to_out_ch_map.size()){
                        s->in_to_out_ch_map.resize(src_idx + 1);
                }

                s->in_to_out_ch_map[src_idx].push_back(dst_idx);

                if(dst_idx >= s->out_ch_contributors.size()){
                        s->out_ch_contributors.resize(dst_idx + 1);
                }

                s->out_ch_contributors[dst_idx]++;
        }

        s->buffer.ch_count = s->out_ch_contributors.size();

        return AF_OK;
};

static af_result_code configure(void *state,
                int in_bps, int in_ch_count, int in_sample_rate)
{
        auto s = static_cast<state_remap *>(state);

        s->buffer.bps = in_bps;
        s->buffer.sample_rate = in_sample_rate;
        s->ch_count = in_ch_count;

        return AF_OK;
}

static void done(void *state){
        auto s = static_cast<state_remap *>(state);

        delete s;
}

static void get_configured_in(void *state,
                int *bps, int *ch_count, int *sample_rate)
{
        auto s = static_cast<state_remap *>(state);

        if(bps) *bps = s->buffer.bps;
        if(ch_count) *ch_count = s->ch_count;
        if(sample_rate) *sample_rate = s->buffer.sample_rate;
}

static void get_configured_out(void *state,
                int *bps, int *ch_count, int *sample_rate)
{
        auto s = static_cast<state_remap *>(state);

        if(bps) *bps = s->buffer.bps;
        if(ch_count) *ch_count = s->buffer.ch_count;
        if(sample_rate) *sample_rate = s->buffer.sample_rate;
}

static void mix_samples(state_remap *s, char *in, char *out){
        int bps = s->buffer.bps;
        double scale = 1.0;
        for(size_t in_ch_idx = 0; in_ch_idx < s->in_to_out_ch_map.size(); in_ch_idx++){
                int32_t in_value = format_from_in_bps(in + bps * in_ch_idx, bps);

                const auto& dst_channels = s->in_to_out_ch_map[in_ch_idx];
                for(int out_ch_idx : dst_channels){
                        char *out_ptr = out + out_ch_idx * bps;
                        int32_t out_value = format_from_in_bps(out_ptr, bps);

                        int32_t new_value = (double)in_value * scale + out_value;

                        format_to_out_bps(out_ptr, bps, new_value);
                }
        }
}

static af_result_code filter(void *state, struct audio_frame **frame){
        auto s = static_cast<state_remap *>(state);

        auto f = *frame;
        if(f->ch_count == 0)
                return AF_OK;

        size_t ch_size = f->data_len / f->ch_count;
        size_t needed_size = s->buffer.ch_count * ch_size;

        if(needed_size > s->data.size()){
                s->data.resize(needed_size, 0);
                s->buffer.data = s->data.data();
                s->buffer.max_size = needed_size;
        }
        s->buffer.data_len = needed_size;

        memset(s->buffer.data, 0, s->buffer.data_len);

        const size_t samples = ch_size / f->bps;
        for(size_t sample = 0; sample < samples; sample++){
                char *in_sample = f->data + sample * f->bps * f->ch_count;
                char *out_sample = s->buffer.data + sample * f->bps * s->buffer.ch_count;

                mix_samples(s, in_sample, out_sample);
        }

        AUDIO_FRAME_DISPOSE(f);
        *frame = &s->buffer;

        return AF_OK;
}

static const struct audio_filter_info audio_filter_remap = {
        .name = "remap",
        .init = init,
        .done = done,
        .configure = configure,
        .get_configured_in = get_configured_in,
        .get_configured_out = get_configured_out,
        .filter = filter,
};

REGISTER_MODULE(remap, &audio_filter_remap, LIBRARY_CLASS_AUDIO_FILTER, AUDIO_FILTER_ABI_VERSION);
