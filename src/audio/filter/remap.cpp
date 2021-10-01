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

#include "debug.h"
#include "audio/audio_filter.h"
#include "audio/types.h"
#include "lib_common.h"


struct state_remap{
        int bps;
        int ch_count;
        int sample_rate;

        struct audio_frame buffer;

        std::vector<std::vector<int>> in_to_out_ch_map;
        std::vector<int> out_ch_contributors;
};

static std::string_view tokenize(std::string_view& str, char delim){
        if(str.empty())
                return {};

        auto token_begin = str.begin();

        while(token_begin != str.end() && *token_begin == delim){
                token_begin++;
        }

        auto token_end = token_begin;

        while(token_end != str.end() && *token_end != delim){
                token_end++;
        }

        str = std::string_view(token_end, str.end() - token_end);

        return std::string_view(token_begin, token_end - token_begin);
}

static int init(const char *cfg, void **state){
        state_remap *s = new state_remap();
        *state = s;

        std::string_view conf(cfg);

        std::string_view tok = tokenize(conf, ',');
        for(; !tok.empty(); tok = tokenize(conf, ',')){
                auto src_t = tokenize(tok, ':');
                auto dst_t = tokenize(tok, ':');
                if(src_t.empty() || dst_t.empty())
                        return -1;

                unsigned src_idx;
                unsigned dst_idx;
                if(std::from_chars(src_t.begin(), src_t.end(), src_idx).ec != std::errc()
                                || std::from_chars(dst_t.begin(), dst_t.end(), dst_idx).ec != std::errc())
                {
                        return -1;
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

        return 0;
};

static af_result_code configure(void *state,
                int in_bps, int in_ch_count, int in_sample_rate)
{
        auto s = static_cast<state_remap *>(state);

        s->bps = in_bps;
        s->ch_count = in_ch_count;
        s->sample_rate = in_sample_rate;

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

        if(bps) *bps = s->bps;
        if(ch_count) *ch_count = s->ch_count;
        if(sample_rate) *sample_rate = s->sample_rate;
}

static void get_configured_out(void *state,
                int *bps, int *ch_count, int *sample_rate)
{
        auto s = static_cast<state_remap *>(state);

        if(bps) *bps = s->bps;
        if(ch_count) *ch_count = s->out_ch_contributors.size();
        if(sample_rate) *sample_rate = s->sample_rate;
}

static af_result_code filter(void *state, struct audio_frame *f){
        auto s = static_cast<state_remap *>(state);

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
