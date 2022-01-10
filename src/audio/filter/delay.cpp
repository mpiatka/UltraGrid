/**
 * @file   delay.cpp
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

#include "debug.h"
#include "audio/audio_filter.h"
#include "audio/types.h"
#include "lib_common.h"
#include "utils/ring_buffer.h"

namespace{
        struct Ring_buf_deleter{
                void operator()(ring_buffer_t *ring){ ring_buffer_destroy(ring); }
        };
}

struct state_delay{
        int samples;
        int bps;
        int ch_count;
        int sample_rate;
        std::unique_ptr<ring_buffer_t, Ring_buf_deleter> ring;
};

static int init(const char *cfg, void **state){
        state_delay *s = new state_delay();
        s->samples = 96000;
        s->bps = 0;
        s->ch_count = 0;

        *state = s;

        return 0;
};

static af_result_code configure(void *state,
                        int in_bps, int in_ch_count, int in_sample_rate)
{
        auto s = static_cast<state_delay *>(state);

        s->bps = in_bps;
        s->ch_count = in_ch_count;
        s->sample_rate = in_sample_rate;

        int delay_size = s->bps * s->ch_count * s->samples;
        s->ring.reset(ring_buffer_init(delay_size * 2));
        ring_fill(s->ring.get(), 0, delay_size);
        return AF_OK;
}

static void done(void *state){
        auto s = static_cast<state_delay *>(state);

        delete s;
}

static void get_configured(void *state,
                        int *bps, int *ch_count, int *sample_rate)
{
        auto s = static_cast<state_delay *>(state);

        if(bps) *bps = s->bps;
        if(ch_count) *ch_count = s->ch_count;
        if(sample_rate) *sample_rate = s->sample_rate;
}

static af_result_code filter(void *state, struct audio_frame *f){
        auto s = static_cast<state_delay *>(state);

        if(f->bps != s->bps || f->ch_count != s->ch_count){
                if(configure(state, f->bps, f->ch_count, f->sample_rate) != AF_OK){
                        return AF_MISCONFIGURED;
                }
        }

        int frame_samples = f->data_len / (f->ch_count * f->bps);

        if(frame_samples <= s->samples){
                ring_buffer_write(s->ring.get(), f->data, f->data_len);
                ring_buffer_read(s->ring.get(), f->data, f->data_len);
        } else {
                int delay_size = s->bps * s->ch_count * s->samples;
                int excess_size = (frame_samples - s->samples) * s->bps * s->ch_count;

                ring_buffer_t *ring = s->ring.get();

                ring_buffer_write(ring,
                                f->data + (f->data_len - delay_size), delay_size);

                memmove(f->data + delay_size, f->data, f->data_len - delay_size);
                ring_buffer_read(s->ring.get(), f->data, delay_size);
        }

        return AF_OK;
}

static const struct audio_filter_info audio_filter_delay = {
        .name = "delay",
        .init = init,
        .done = done,
        .configure = configure,
        .get_configured_in = get_configured,
        .get_configured_out = get_configured,
        .filter = filter,
};

REGISTER_MODULE(delay, &audio_filter_delay, LIBRARY_CLASS_AUDIO_FILTER, AUDIO_FILTER_ABI_VERSION);
