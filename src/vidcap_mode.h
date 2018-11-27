/**
 * @file   vidcap_mode.h
 * @author Martin Piatka <445597@mail.muni.cz>
 *
 * @ingroup vidcap
 */
/**
 * Copyright (c) 2005-2018 CESNET z.s.p.o
 * Copyright (c) 2002 University of Southern California
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
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 * 
 *      This product includes software developed by the University of Southern
 *      California Information Sciences Institute. This product also includes
 *      software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the University, Institute, CESNET nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
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
 *
 */

#ifndef VIDCAP_MODE_H
#define VIDCAP_MODE_H

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

struct vidcap_mode {
        int mode_num;

        enum {
                Frame_size_dicrete,
                Frame_size_stepwise,
                Frame_size_cont,
        } frame_size_type;

        union {
                struct {
                        int width;
                        int height;
                } discrete;
                struct {
                        int min_width;
                        int max_width;
                        int min_height;
                        int max_height;
                        int step_width;
                        int step_height;
                } stepwise;
        } frame_size;

        enum {
                Fps_unknown,
                Fps_discrete,
                Fps_stepwise,
                Fps_cont,
        } fps_type;

        union {

                struct {
                        long long numerator;
                        int denominator;
                } fraction;

                struct {
                        long long min_numerator;
                        int min_denominator;
                        long long max_numerator;
                        int max_denominator;
                        long long step_numerator;
                        int step_denominator;
                } stepwise;
        } fps;

        char format[32];
        char format_desc[32];
};

#ifdef __cplusplus
}
#endif // __cplusplus

#endif
