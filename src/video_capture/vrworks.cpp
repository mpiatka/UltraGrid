/**
 * @file   video_capture/vrworks.c
 * @author Martin Piatka    <piatka@cesnet.cz>
 *         Martin Pulec     <pulec@cesnet.cz>
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
#include <cmath>
#include <nvss_video.h>
#include <nvstitch_common.h>
#include <nvstitch_common_video.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <xml_util/xml_utility_video.h>

#define PI 3.14159265

const char *log_str = "[vrworks] ";

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

        nvstitchStitcherPipelineType pipeline;
        nvstitchStitcherQuality quality;
        unsigned width;
        nvstitchRect_t roi;

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

static void calculate_roi(vidcap_vrworks_state *s){
        double left = 0;
        double right = 0;
        double top = 0;
        double bottom = 0;
        for(const auto &cam : s->cam_properties){
                double vfov_half = std::atan((cam.image_size.x / 2) / cam.focal_length) * 180 / PI;
                double hfov_half = std::atan((cam.image_size.y / 2) / cam.focal_length) * 180 / PI;

                float r31 = cam.extrinsics.rotation[6];
                float r32 = cam.extrinsics.rotation[7];
                float r33 = cam.extrinsics.rotation[8];
                double yaw = std::atan2(-r31, std::sqrt(r32*r32 + r33*r33)) * 180 / PI;
                double pitch = std::atan2(r32, r33) * 180 / PI;
                std::cout << vfov_half << " "
                        << hfov_half << " "
                        << yaw << " "
                        << pitch << std::endl;

                double left_edge = yaw - vfov_half;
                double right_edge = yaw + vfov_half;
                double top_edge = pitch - hfov_half;
                double bottom_edge = pitch + hfov_half;
                if(left_edge < left) left = left_edge;
                if(right_edge > right) right = right_edge;
                if(bottom_edge > bottom) bottom = bottom_edge;
                if(top_edge < top) top = top_edge;
        }

        double px_per_deg = s->width / 360.0;

        unsigned center_x = s->width / 2;
        unsigned center_y = s->width / 4;

        unsigned roi_left = center_x + left* px_per_deg;
        unsigned roi_right = center_x + right* px_per_deg;
        unsigned roi_top = center_y + top * px_per_deg;
        unsigned roi_bottom = center_y + bottom* px_per_deg;

        log_msg(LOG_LEVEL_INFO, "[vrworks cap.] Calculated roi: %u,%u %ux%u\n",
                        roi_left,
                        roi_top,
                        roi_right - roi_left,
                        roi_bottom - roi_top);
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
        s->stitcher_properties.pano_width = s->width;
        s->stitcher_properties.quality = s->quality;
        s->stitcher_properties.num_gpus = 1;
        s->stitcher_properties.ptr_gpus = &selected_gpu;
        s->stitcher_properties.pipeline = s->pipeline;
        s->stitcher_properties.projection = NVSTITCH_PANORAMA_PROJECTION_EQUIRECTANGULAR;
        s->stitcher_properties.output_roi = s->roi;
        //s->stitcher_properties.mono_flags = NVSTITCH_MONO_FLAGS_ENABLE_DEPTH_ALIGNMENT;

        // Fetch rig parameters from XML file.
        if (!xmlutil::readCameraRigXml("rig_spec.xml", s->cam_properties, &s->rig_properties))
        {
                std::cerr << log_str << "Failed to retrieve rig paramters from XML file." << std::endl;
                return false;
        }

        calculate_roi(s);

        nvstitchResult res;
        res = nvssVideoCreateInstance(&s->stitcher_properties, &s->rig_properties, &s->stitcher);
        if(res != NVSTITCH_SUCCESS){
                std::cout << log_str << "Failed to create stitcher instance." << std::endl;
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
                std::cerr << log_str << "Failed to allocate tmpframe" << std::endl;
                return false;
        }

        return true;
}

static void parse_fmt(vidcap_vrworks_state *s, const char * const fmt){
        s->pipeline = NVSTITCH_STITCHER_PIPELINE_MONO_EQ;
        s->quality = NVSTITCH_STITCHER_QUALITY_HIGH;
        s->width = 3840;
        s->roi = nvstitchRect_t{ 0, 0, 0, 0 };
        //s->roi = nvstitchRect_t{ 900, 700, 2100, 500 };

        if(!fmt)
                return;

        const nvstitchStitcherQuality quality_vals[] = {
                NVSTITCH_STITCHER_QUALITY_LOW,
                NVSTITCH_STITCHER_QUALITY_MEDIUM,
                NVSTITCH_STITCHER_QUALITY_HIGH,
        };

        const nvstitchStitcherPipelineType pipelines[] = {
                NVSTITCH_STITCHER_PIPELINE_MONO,
                NVSTITCH_STITCHER_PIPELINE_MONO_EQ,
        };

        char *tmp = strdup(fmt);
        char *init_fmt = tmp;
        char *save_ptr = NULL;
        char *item;
#define FMT_CMP(param) (strncmp(item, (param), strlen((param))) == 0)
        while((item = strtok_r(init_fmt, ":", &save_ptr))) {
                if (FMT_CMP("width=")) {
                        s->width = atoi(strchr(item, '=') + 1);
                } else if(FMT_CMP("roi=")){
                        unsigned left, top, width, height;
                        int count = sscanf(strchr(item, '=') + 1, "%u,%u,%u,%u",
                                        &left, &top, &width, &height); 
                        if(count == 4){
                                s->roi = nvstitchRect_t{left, top, width, height};
                        }
                } else if(FMT_CMP("quality=")){
                        unsigned quality = atoi(strchr(item, '=') + 1);
                        if(quality < sizeof(quality_vals) / sizeof(*quality_vals)){
                                s->quality = quality_vals[quality];
                        }
                } else if(FMT_CMP("pipeline=")){
                        unsigned pipeline = atoi(strchr(item, '=') + 1);
                        if(pipeline < sizeof(pipelines) / sizeof(*pipelines)){
                                s->pipeline = pipelines[pipeline];
                        }
                }
                init_fmt = NULL;
        }
}

static int
vidcap_vrworks_init(struct vidcap_params *params, void **state)
{
        struct vidcap_vrworks_state *s;

        printf("vidcap_vrworks_init\n");


        s = new vidcap_vrworks_state();
        if(s == NULL) {
                printf("Unable to allocate vrworks capture state\n");
                return VIDCAP_INIT_FAIL;
        }

        s->audio_source_index = -1;
        s->frames = 0;
        gettimeofday(&s->t0, NULL);

        if(vidcap_params_get_fmt(params) && strcmp(vidcap_params_get_fmt(params), "help") == 0) {
               show_help(); 
               free(s);
               return VIDCAP_INIT_NOERR;
        }

        parse_fmt(s, vidcap_params_get_fmt(params));

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

        if(!s) return;

        for (int i = 0; i < s->devices_cnt; ++i) {
                VIDEO_FRAME_DISPOSE(s->captured_frames[i]);
                vidcap_done(s->devices[i]);
        }
        free(s->captured_frames);
        free(s->devices);
        
        vf_free(s->frame);
        cudaFreeHost(s->tmpframe);

        nvssVideoDestroyInstance(s->stitcher);

        delete s;
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

        for(unsigned i = 0; i < in_frame->tiles[0].height; i++){
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

static void result_data_delete(struct video_frame *buf){
        if(!buf)
                return;

        cudaFreeHost(buf->tiles[0].data);
}

static bool allocate_result_frame(vidcap_vrworks_state *s, unsigned width, unsigned height){
        video_desc desc{};
        desc.width = width;
        desc.height = height;
        desc.color_spec = RGBA;
        desc.tile_count = 1;
        desc.fps = 30;

        s->frame = vf_alloc_desc(desc);
        s->frame->tiles[0].data_len = vc_get_linesize(desc.width,
                        desc.color_spec) * desc.height;

        if(cudaMallocHost(&s->frame->tiles[0].data, s->frame->tiles[0].data_len) != cudaSuccess){
                std::cerr << log_str << "Failed to allocate result frame" << std::endl;
                return false;
        }

        s->frame->callbacks.data_deleter = result_data_delete;
        s->frame->callbacks.recycle = NULL;

        return true;
}

static bool download_stitched(vidcap_vrworks_state *s){
        nvstitchImageBuffer_t output_image;

        nvstitchResult res;
        res = nvssVideoGetOutputBuffer(s->stitcher, NVSTITCH_EYE_MONO, &output_image);
        if(res != NVSTITCH_SUCCESS){
                std::cerr << log_str << "Failed to get output buffer" << std::endl;
                return false;
        }

        if(!s->frame){
                if(!allocate_result_frame(s, output_image.width, output_image.height)){
                        return false;
                }
        }
        cudaStream_t out_stream = nullptr;
        res = nvssVideoGetOutputStream(s->stitcher, NVSTITCH_EYE_MONO, &out_stream);
        if(res != NVSTITCH_SUCCESS){
                std::cerr << log_str << "Failed to get output stream" << std::endl;
                return false;
        }

        if (cudaMemcpy2DAsync(s->frame->tiles[0].data, output_image.row_bytes,
                                output_image.dev_ptr, output_image.pitch,
                                output_image.row_bytes, output_image.height,
                                cudaMemcpyDeviceToHost, out_stream) != cudaSuccess)
        {
                std::cerr << log_str << "Error copying output panorama from CUDA buffer" << std::endl;
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

static bool check_in_format(vidcap_vrworks_state *s, video_frame *in, int i){
        if(in->tile_count != 1){
                std::cerr << log_str << "Only frames with tile_count == 1 are supported" << std::endl;
                return false;
        }

        unsigned int expected_w = s->cam_properties[i].image_size.x;
        unsigned int expected_h = s->cam_properties[i].image_size.y;

        if(in->tiles[0].width != expected_w
                        || in->tiles[0].height != expected_h)
        {
                std::cerr << log_str << "Wrong resolution for input " << i << "!"
                       << " Expected " << expected_w << "x" << expected_h
                       << ", but got " << in->tiles[0].width << "x" << in->tiles[0].height << std::endl;
                return false;
        }

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
                if (!check_in_format(s, frame, i)) {
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

                nvstitchOverlap_t overlap;
                nvstitchSeam_t seam;
                unsigned overlap_count;
                nvssVideoGetOverlapCount(s->stitcher, &overlap_count);
                log_msg(LOG_LEVEL_INFO, "[vrworks cap.] %u overlaps\n", overlap_count);

                for(unsigned i = 0; i < overlap_count; i++){
                        nvssVideoGetOverlapInfo(s->stitcher, i, &overlap, &seam);
                        log_msg(LOG_LEVEL_INFO, "overlap(%u,%u): %ux%u %u,%u\n",
                                        overlap.camera_left,
                                        overlap.camera_right,
                                        overlap.overlap_rect.left,
                                        overlap.overlap_rect.top,
                                        overlap.overlap_rect.width,
                                        overlap.overlap_rect.height
                               );

                        log_msg(LOG_LEVEL_INFO, "seam %d ", seam.reproj_width);
                        switch(seam.seam_type){
                                case NVSTITCH_SEAM_TYPE_VERTICAL:
                                        log_msg(LOG_LEVEL_INFO, "(vertical) %u\n", seam.properties.vertical.x_offset);
                                        break;
                                case NVSTITCH_SEAM_TYPE_HORIZONTAL:
                                        log_msg(LOG_LEVEL_INFO, "(horizontal) %u\n", seam.properties.horizontal.y_offset);
                                        break;
                                case NVSTITCH_SEAM_TYPE_DIAGONAL:
                                        log_msg(LOG_LEVEL_INFO, "(diagonal) %u,%u %u,%u\n",
                                                        seam.properties.diagonal.p1.x,
                                                        seam.properties.diagonal.p1.y,
                                                        seam.properties.diagonal.p2.x,
                                                        seam.properties.diagonal.p2.y
                                                        );
                                        break;

                        }
                }
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

