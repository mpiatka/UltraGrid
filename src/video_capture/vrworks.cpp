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

        unsigned char *tmpframe;
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

static bool init_stitcher(struct vidcap_vrworks_state *s){
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
                return false;
        }

        nvstitchResult res;
        res = nvssVideoCreateInstance(&s->stitcher_properties, &s->rig_properties, &s->stitcher);
        if(res != NVSTITCH_SUCCESS){
                std::cout << std::endl << "Failed to create stitcher instance." << std::endl;
                return false;
        }

        return true;
}

static bool alloc_tmp_frame(struct vidcap_vrworks_state *s){
        //Make sure every input can fit
        size_t max_size = 0;
        for(const auto &cam : s->cam_properties){
                if(cam.image_size.x * cam.image_size.y > max_size){
                        max_size = cam.image_size.x * cam.image_size.y;
                }
        }

        const int bpp = 4; //RGBA
        if(cudaMallocHost(&s->tmpframe, max_size*bpp) != cudaSuccess){
                std::cout << std::endl << "Failed to allocate tmpframe" << std::endl;
                return false;
        }

        return true;
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

        s->frame = nullptr;

        if(!init_stitcher(s)){
                goto error;
        }

        if(!alloc_tmp_frame(s)){
                goto error;
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

static bool upload_frame(vidcap_vrworks_state *s, video_frame *in_frame, int i){
        auto decoder = get_decoder_from_to(in_frame->color_spec, RGBA, false);
        if(!decoder){
                std::cerr << "Failed to get conversion function to RGBA" << std::endl;
                return false;
        }

        unsigned int width = in_frame->tiles[0].width;
        unsigned int height = in_frame->tiles[0].height;
        int in_line_size = vc_get_linesize(width, in_frame->color_spec);
        int out_line_size = vc_get_linesize(width, RGBA);

        for(int i = 0; i < in_frame->tiles[0].height; i++){
                decoder(s->tmpframe + out_line_size * i,
                                (const unsigned char *)(in_frame->tiles[0].data + in_line_size * i),
                                out_line_size,
                                0, 8, 16);
        }

        nvstitchImageBuffer_t input_image;
        nvstitchResult res;
        res = nvssVideoGetInputBuffer(s->stitcher, i, &input_image);
        if(res != NVSTITCH_SUCCESS){
                std::cerr << std::endl << "Failed to get input buffer." << std::endl;
                return false;
        }
        size_t row_bytes = out_line_size;
        if (cudaMemcpy2D(input_image.dev_ptr, input_image.pitch,
                                s->tmpframe, row_bytes,
                                row_bytes, height,
                                cudaMemcpyHostToDevice) != cudaSuccess)
        {
                std::cerr << "Error copying RGBA image bitmap to CUDA buffer" << std::endl;
                return false;
        }

        return true;
}

static bool download_stitched(vidcap_vrworks_state *s){
        nvstitchImageBuffer_t output_image;

        nvstitchResult res;
        res = nvssVideoGetOutputBuffer(s->stitcher, NVSTITCH_EYE_MONO, &output_image);
        //TODO

        if(!s->frame){
                video_desc desc{};
                desc.width = output_image.width;
                desc.height = output_image.height;
                desc.color_spec = RGBA;
                desc.tile_count = 1;
                desc.fps = 30;

                s->frame = vf_alloc_desc_data(desc);
        }
        cudaStream_t out_stream = nullptr;
        res = nvssVideoGetOutputStream(s->stitcher, NVSTITCH_EYE_MONO, &out_stream);
        //TODO

        if (cudaMemcpy2DAsync(s->frame->tiles[0].data, output_image.row_bytes,
                                output_image.dev_ptr, output_image.pitch,
                                output_image.row_bytes, output_image.height,
                                cudaMemcpyDeviceToHost, out_stream) != cudaSuccess)
        {
                std::cout << "Error copying output panorama from CUDA buffer" << std::endl;
                return false;
        }
#if 0
        if (cudaStreamSynchronize(out_stream) != cudaSuccess)
        {
                std::cerr << "Error synchronizing with the output CUDA stream" << std::endl;
                return false;
        }
#endif

        return true;
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

        nvstitchResult res;

        for (int i = 0; i < s->devices_cnt; ++i) {
                frame = NULL;
                while(!frame) {
                        frame = vidcap_grab(s->devices[i], &audio_frame);
                }
                if (s->audio_source_index == -1 && audio_frame != NULL) {
                        fprintf(stderr, "[vrworks] Locking device #%d as an audio source.\n",
                                        i);
                        s->audio_source_index = i;
                }
                if (s->audio_source_index == i) {
                        *audio = audio_frame;
                }
                if (false) {
                        fprintf(stderr, "[vrworks] Different format detected: ");
                        fprintf(stderr, "\n");
                        
                        return NULL;
                }

                if(!upload_frame(s, frame, i)){
                        return NULL;
                }

                s->captured_frames[i] = frame;
        }

        res = nvssVideoStitch(s->stitcher);
        if(res != NVSTITCH_SUCCESS){
                std::cout << std::endl << "Failed to stitch." << std::endl;
                return NULL;
        }

        if(!download_stitched(s)){
                return NULL;
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

