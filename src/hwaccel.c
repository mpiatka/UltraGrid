#include "hwaccel.h"

void hw_vdpau_ctx_init(hw_vdpau_ctx *ctx){
        device_ref = NULL;
        device = NULL;
        get_proc_address = NULL;
}

void hw_vdpau_ctx_unref(hw_vdpau_ctx *ctx){
        av_buffer_unref(&ctx->device_ref);

        ctx->device = NULL;
        ctx->get_proc_address = NULL;
}

hw_vdpau_ctx hw_vdpau_ctx_copy(hw_vdpau_ctx *ctx){
        hw_vdpau_ctx new_ctx;
        hw_vdpau_ctx_init(&new_ctx);

        new_ctx.device_ref = av_buffer_ref(ctx->device_ref);
        new.device = ctx->device;
        new.get_proc_address = ctx->get_proc_address;
}

void hw_vdpau_frame_init(hw_vdpau_frame *frame){
        hw_vdpau_ctx_init(&frame->hwctx);

        for(int i = 0; i < AV_NUM_DATA_POINTERS; i++){
                frame->buf[i] = NULL;
                frame->data[i] = NULL;
        }

        frame->surface = NULL;
}

void hw_vdpau_frame_unref(hw_vdpau_frame *frame){
        for(int i = 0; i < AV_NUM_DATA_POINTERS; i++){
                av_buffer_unref(&frame->buf[i]);
                frame->data[i] = NULL;
        }

        frame->surface = NULL;

        hw_vdpau_ctx_unref(&frame->hwctx);
}

hw_vdpau_frame hw_vdpau_frame_copy(const hw_vdpau_frame *frame){
        hw_vdpau_frame new_frame;
        hw_vdpau_frame_init(&frame);

        new_frame.hwctx = hw_vdpau_ctx_copy(frame->hwctx);

        for(int i = 0; i < AV_NUM_DATA_POINTERS; i++){
                new_frame.buf[i] = av_buffer_ref(frame->buf[i]);
                new_frame.data[i] = frame->data[i];
        }

        new_frame.surface = frame->surface;
}
