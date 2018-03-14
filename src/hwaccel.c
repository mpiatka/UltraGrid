#include <assert.h>
#include "hwaccel.h"

void hw_vdpau_ctx_init(hw_vdpau_ctx *ctx){
        ctx->device_ref = NULL;
        ctx->device = 0;
        ctx->get_proc_address = NULL;
}

void hw_vdpau_ctx_unref(hw_vdpau_ctx *ctx){
        av_buffer_unref(&ctx->device_ref);

        hw_vdpau_ctx_init(ctx);
}

hw_vdpau_ctx hw_vdpau_ctx_copy(const hw_vdpau_ctx *ctx){
        hw_vdpau_ctx new_ctx;
        hw_vdpau_ctx_init(&new_ctx);

        new_ctx.device_ref = av_buffer_ref(ctx->device_ref);
        new_ctx.device = ctx->device;
        new_ctx.get_proc_address = ctx->get_proc_address;

        return new_ctx;
}

void hw_vdpau_frame_init(hw_vdpau_frame *frame){
        hw_vdpau_ctx_init(&frame->hwctx);

        for(int i = 0; i < AV_NUM_DATA_POINTERS; i++){
                frame->buf[i] = NULL;
                frame->data[i] = NULL;
        }

        frame->surface = 0;
}

void hw_vdpau_frame_unref(hw_vdpau_frame *frame){
        for(int i = 0; i < AV_NUM_DATA_POINTERS; i++){
                av_buffer_unref(&frame->buf[i]);
        }

        hw_vdpau_ctx_unref(&frame->hwctx);

        hw_vdpau_frame_init(frame);
}

void hw_vdpau_free_extra_data(void *frame){
        hw_vdpau_frame_unref((hw_vdpau_frame *) frame);
}

hw_vdpau_frame hw_vdpau_frame_copy(const hw_vdpau_frame *frame){
        hw_vdpau_frame new_frame;
        hw_vdpau_frame_init(&new_frame);

        new_frame.hwctx = hw_vdpau_ctx_copy(&frame->hwctx);

        for(int i = 0; i < AV_NUM_DATA_POINTERS; i++){
                if(frame->buf[i])
                        new_frame.buf[i] = av_buffer_ref(frame->buf[i]);
                new_frame.data[i] = frame->data[i];
        }

        new_frame.surface = frame->surface;

        return new_frame;
}

void *hw_vdpau_frame_data_cpy(void *dst, const void *src, size_t n){
        assert(n == sizeof(hw_vdpau_frame));

        hw_vdpau_frame *new = (hw_vdpau_frame *) dst;

        *new = hw_vdpau_frame_copy((const hw_vdpau_frame *) src);

        return new;
}

hw_vdpau_frame *hw_vdpau_frame_from_avframe(hw_vdpau_frame *dst, const AVFrame *src){
        hw_vdpau_frame_init(dst);

        AVHWFramesContext *frame_ctx = (AVHWFramesContext *) src->hw_frames_ctx->data;
        AVHWDeviceContext *device_ctx = frame_ctx->device_ctx; 
        AVVDPAUDeviceContext *vdpau_ctx = (AVVDPAUDeviceContext *) device_ctx->hwctx;


        dst->hwctx.device_ref = av_buffer_ref(frame_ctx->device_ref);
        dst->hwctx.device = vdpau_ctx->device;
        dst->hwctx.get_proc_address = vdpau_ctx->get_proc_address;

        for(int i = 0; i < AV_NUM_DATA_POINTERS; i++){
                if(src->buf[i])
                        dst->buf[i] = av_buffer_ref(src->buf[i]);
                dst->data[i] = src->data[i];
        }

        dst->surface = (VdpVideoSurface) dst->data[3];

        return dst;
}

void vdp_funcs_init(vdp_funcs *f){
        memset(f, 0, sizeof(vdp_funcs));
}

void vdp_funcs_load(vdp_funcs *f, VdpDevice device, VdpGetProcAddress *get_proc_address){
        get_proc_address(device, VDP_FUNC_ID_VIDEO_SURFACE_GET_PARAMETERS, &f->videoSurfaceGetParameters);
        get_proc_address(device, VDP_FUNC_ID_VIDEO_MIXER_CREATE, &f->videoMixerCreate);
        get_proc_address(device, VDP_FUNC_ID_VIDEO_MIXER_DESTROY, &f->videoMixerDestroy);
        get_proc_address(device, VDP_FUNC_ID_VIDEO_MIXER_RENDER, &f->videoMixerRender);

        get_proc_address(device, VDP_FUNC_ID_OUTPUT_SURFACE_CREATE, &f->outputSurfaceCreate);
        get_proc_address(device, VDP_FUNC_ID_OUTPUT_SURFACE_DESTROY, &f->outputSurfaceDestroy);
        get_proc_address(device, VDP_FUNC_ID_OUTPUT_SURFACE_GET_PARAMETERS, &f->outputSurfaceGetParameters);

        get_proc_address(device, VDP_FUNC_ID_GET_ERROR_STRING, &f->getErrorString);
}
