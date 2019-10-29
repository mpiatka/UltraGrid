/**
 * @file   video_capture/vrworks.c
 * @author Martin Piatka <piatka@cesnet.cz>
 *
 * @brief Vrworks 360 video stitcher
 */
/*
 * Copyright (c) 2019 CESNET z.s.p.o.
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
#include "host.h"
#include "lib_common.h"
#include "video.h"
#include "video_capture.h"

#include "tv.h"

#include "audio/audio.h"

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <nvss_video.h>
#include <nvstitch_common.h>
#include <nvstitch_common_video.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <xml_util/xml_utility_video.h>

/* prototypes of functions defined in this module */
static void show_help(void);

static void show_help()
{
        printf("Vrworks 360 video stitcher\n");
        printf("Usage\n");
        printf("\t-t vrworks -t <dev1_config> -t <dev2_config> ....]\n");
        printf("\t\twhere devn_config is a complete configuration string of device involved in stitching\n");

}

struct vidcap_vrworks_state {
        struct vidcap     **devices;
        int                 devices_cnt;

        struct video_frame      **captured_frames;
        struct video_frame       *frame; 
        int frames;
        struct       timeval t, t0;

        int          audio_source_index;
        nvssVideoHandle stitcher;
		nvssVideoStitcherProperties_t stitcher_properties;
		std::vector<nvstitchCameraProperties_t> cam_properties;
		nvstitchVideoRigProperties_t rig_properties;
};


static struct vidcap_type *
vidcap_vrworks_probe(bool verbose, void (**deleter)(void *))
{
        UNUSED(verbose);
        *deleter = free;
	struct vidcap_type*		vt;
    
	vt = (struct vidcap_type *) calloc(1, sizeof(struct vidcap_type));
	if (vt != NULL) {
		vt->name        = "vrworks";
		vt->description = "vrworks video capture";
	}
	return vt;
}

static int
vidcap_vrworks_init(struct vidcap_params *params, void **state)
{
	struct vidcap_vrworks_state *s;

	printf("vidcap_vrworks_init\n");


        s = (struct vidcap_vrworks_state *) calloc(1, sizeof(struct vidcap_vrworks_state));
	if(s == NULL) {
		printf("Unable to allocate vrworks capture state\n");
		return VIDCAP_INIT_FAIL;
	}

        s->audio_source_index = -1;
        s->frames = 0;
        gettimeofday(&s->t0, NULL);

        if(vidcap_params_get_fmt(params) && strcmp(vidcap_params_get_fmt(params), "") != 0) {
                show_help();
                free(s);
                return VIDCAP_INIT_NOERR;
        }


        s->devices_cnt = 0;
        struct vidcap_params *tmp = params;
        while((tmp = vidcap_params_get_next(tmp))) {
                if (vidcap_params_get_driver(tmp) != NULL)
                        s->devices_cnt++;
                else
                        break;
        }

        s->devices = (vidcap **) calloc(s->devices_cnt, sizeof(struct vidcap *));
        tmp = params;
        for (int i = 0; i < s->devices_cnt; ++i) {
                tmp = vidcap_params_get_next(tmp);

                int ret = initialize_video_capture(NULL, (struct vidcap_params *) tmp, &s->devices[i]);
                if(ret != 0) {
                        fprintf(stderr, "[vrworks] Unable to initialize device %d (%s:%s).\n",
                                        i, vidcap_params_get_driver(tmp),
                                        vidcap_params_get_fmt(tmp));
                        goto error;
                }
        }

        s->captured_frames = (struct video_frame **) calloc(s->devices_cnt, sizeof(struct video_frame *));

        s->frame = vf_alloc(s->devices_cnt);

		int num_gpus;
		cudaGetDeviceCount(&num_gpus);

		int selected_gpu;

		for (int gpu = 0; gpu < num_gpus; ++gpu)
		{
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, gpu);

			// Require minimum compute 5.2
			if (prop.major > 5 || (prop.major == 5 && prop.minor >= 2))
			{
				selected_gpu = gpu;
				break;
			}
		}

		s->stitcher_properties.version = NVSTITCH_VERSION;
		s->stitcher_properties.pano_width = 3840;
		s->stitcher_properties.quality = NVSTITCH_STITCHER_QUALITY_HIGH;
		s->stitcher_properties.num_gpus = 1;
		s->stitcher_properties.ptr_gpus = &selected_gpu;
		s->stitcher_properties.pipeline = NVSTITCH_STITCHER_PIPELINE_MONO;
		s->stitcher_properties.projection = NVSTITCH_PANORAMA_PROJECTION_EQUIRECTANGULAR;
		s->stitcher_properties.output_roi = nvstitchRect_t{ 0, 0, 0, 0 };

		// Fetch rig parameters from XML file.
		if (!xmlutil::readCameraRigXml("rig_spec.xml", s->cam_properties, &s->rig_properties))
		{
			std::cout << std::endl << "Failed to retrieve rig paramters from XML file." << std::endl;
			return 1;
		}

		nvssVideoHandle stitcher;
		nvstitchResult res;
		res = nvssVideoCreateInstance(&s->stitcher_properties, &s->rig_properties, &s->stitcher);
		if(res != NVSTITCH_SUCCESS){
			std::cout << std::endl << "Failed to create stitcher instance." << std::endl;
			return 1;
		}
        
        *state = s;
		return VIDCAP_INIT_OK;

error:
        if(s->devices) {
                int i;
                for (i = 0u; i < s->devices_cnt; ++i) {
                        if(s->devices[i]) {
                                 vidcap_done(s->devices[i]);
                        }
                }
        }
        free(s);
        return VIDCAP_INIT_FAIL;
}

static void
vidcap_vrworks_done(void *state)
{
	struct vidcap_vrworks_state *s = (struct vidcap_vrworks_state *) state;

	assert(s != NULL);

	if (s != NULL) {
                int i;
		for (i = 0; i < s->devices_cnt; ++i) {
                         vidcap_done(s->devices[i]);
		}
	}
        
        vf_free(s->frame);
}

static struct video_frame *
vidcap_vrworks_grab(void *state, struct audio_frame **audio)
{
	struct vidcap_vrworks_state *s = (struct vidcap_vrworks_state *) state;
        struct audio_frame *audio_frame = NULL;
        struct video_frame *frame = NULL;

        for (int i = 0; i < s->devices_cnt; ++i) {
                VIDEO_FRAME_DISPOSE(s->captured_frames[i]);
        }

        *audio = NULL;

        for (int i = 0; i < s->devices_cnt; ++i) {
                frame = NULL;
                while(!frame) {
                        frame = vidcap_grab(s->devices[i], &audio_frame);
                }
                if (i == 0) {
                        s->frame->color_spec = frame->color_spec;
                        s->frame->interlacing = frame->interlacing;
                        s->frame->fps = frame->fps;
                }
                if (s->audio_source_index == -1 && audio_frame != NULL) {
                        fprintf(stderr, "[vrworks] Locking device #%d as an audio source.\n",
                                        i);
                        s->audio_source_index = i;
                }
                if (s->audio_source_index == i) {
                        *audio = audio_frame;
                }
                if (frame->color_spec != s->frame->color_spec ||
                                frame->fps != s->frame->fps ||
                                frame->interlacing != s->frame->interlacing) {
                        fprintf(stderr, "[vrworks] Different format detected: ");
                        if(frame->color_spec != s->frame->color_spec)
                                fprintf(stderr, "codec");
                        if(frame->interlacing != s->frame->interlacing)
                                fprintf(stderr, "interlacing");
                        if(frame->fps != s->frame->fps)
                                fprintf(stderr, "FPS (%.2f and %.2f)", frame->fps, s->frame->fps);
                        fprintf(stderr, "\n");
                        
                        return NULL;
                }
                vf_get_tile(s->frame, i)->width = vf_get_tile(frame, 0)->width;
                vf_get_tile(s->frame, i)->height = vf_get_tile(frame, 0)->height;
                vf_get_tile(s->frame, i)->data_len = vf_get_tile(frame, 0)->data_len;
                vf_get_tile(s->frame, i)->data = vf_get_tile(frame, 0)->data;
                s->captured_frames[i] = frame;
        }
        s->frames++;
        gettimeofday(&s->t, NULL);
        double seconds = tv_diff(s->t, s->t0);    
        if (seconds >= 5) {
            float fps  = s->frames / seconds;
            log_msg(LOG_LEVEL_INFO, "[vrworks cap.] %d frames in %g seconds = %g FPS\n", s->frames, seconds, fps);
            s->t0 = s->t;
            s->frames = 0;
        }  

	return s->frame;
}

static const struct video_capture_info vidcap_vrworks_info = {
        vidcap_vrworks_probe,
        vidcap_vrworks_init,
        vidcap_vrworks_done,
        vidcap_vrworks_grab,
};

REGISTER_MODULE(vrworks, &vidcap_vrworks_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

