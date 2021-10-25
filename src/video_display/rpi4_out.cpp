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

struct rpi4_display_state{
        struct video_frame *f = nullptr;

        struct video_desc current_desc;
};

static void *display_rpi4_init(struct module *parent, const char *cfg, unsigned int flags)
{
        rpi4_display_state *s = new rpi4_display_state();
        return s;
}

static void display_rpi4_done(void *state) {
        auto *s = static_cast<rpi4_display_state *>(state);

        vf_free(s->f);

        delete s;
}

static struct video_frame *display_rpi4_getf(void *state) {
        auto *s = static_cast<rpi4_display_state *>(state);

        return s->f;
}

static int display_rpi4_putf(void *state, struct video_frame *frame, int flags)
{
        vf_recycle(frame);
}

static void display_rpi4_run(void *)
{
}

static int display_rpi4_reconfigure(void *state, struct video_desc desc)
{
        auto *s = static_cast<rpi4_display_state *>(state);

        assert(desc.color_spec == RPI4_8);

        s->current_desc = desc;
        vf_free(s->f);
        s->f= nullptr;

        s->f = vf_alloc_desc_data(desc);

        memset(s->f->tiles[0].data, 0, s->f->tiles[0].data_len);

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
