#ifndef HWACCEL_H
#define HWACCEL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vdpau.h>
#include <libavutil/hwcontext_vaapi.h>
#include <libavcodec/vdpau.h>
#include <libavcodec/vaapi.h>

typedef struct hw_vdpau_ctx{
        AVBufferRef *device_ref; //Av codec buffer reference

        //These are just pointers to the device_ref buffer
        VdpDevice device;
        VdpGetProcAddress *get_proc_address;
} hw_vdpau_ctx;

typedef struct hw_vdpau_frame{
        hw_vdpau_ctx hwctx;
        AVBufferRef *buf[AV_NUM_DATA_POINTERS];

        //These are just pointers to the buffer
        uint8_t *data[AV_NUM_DATA_POINTERS];
        VdpVideoSurface surface; // Same as data[3]
} hw_vdpau_frame;

void hw_vdpau_ctx_init(hw_vdpau_ctx *ctx);
void hw_vdpau_ctx_unref(hw_vdpau_ctx *ctx);
hw_vdpau_ctx hw_vdpau_ctx_copy(const hw_vdpau_ctx *ctx);

void hw_vdpau_frame_init(hw_vdpau_frame *frame);
void hw_vdpau_frame_unref(hw_vdpau_frame *frame);
void hw_vdpau_free_extra_data(void *frame);
hw_vdpau_frame hw_vdpau_frame_copy(const hw_vdpau_frame *frame);

void *hw_vdpau_frame_data_cpy(void *dst, const void *src, size_t n);

hw_vdpau_frame *hw_vdpau_frame_from_avframe(hw_vdpau_frame *dst, const AVFrame *src);

typedef struct vdp_funcs{
        VdpVideoSurfaceGetParameters *videoSurfaceGetParameters;
} vdp_funcs;

void vdp_funcs_init(vdp_funcs *);
void vdp_funcs_load(vdp_funcs *, VdpDevice, VdpGetProcAddress *);

#ifdef __cplusplus
}
#endif

#endif
