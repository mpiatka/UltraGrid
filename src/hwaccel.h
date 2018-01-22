#ifndef HWACCEL_H
#define HWACCEL_H

#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vdpau.h>
#include <libavutil/hwcontext_vaapi.h>
#include <libavcodec/vdpau.h>
#include <libavcodec/vaapi.h>

struct hw_vdpau_ctx{
        AVBufferRef *device_ref; //Av codec buffer reference

        //Theese are just pointers to the device_ref buffer
        VdpDevice *device;
        VdpGetProcAddress *get_proc_address;
};

struct hw_vdpau_frame{
        hw_vdpau_ctx hwctx;
        AVBufferRef *buf[AV_NUM_DATA_POINTERS];

        //Theese are just pointers to the buffer
        uint8_t *data[AV_NUM_DATA_POINTERS];
        VdpVideoSurface *surface; // Same as data[3]
};

void hw_vdpau_ctx_init(hw_vdpau_ctx *ctx);
void hw_vdpau_ctx_unref(hw_vdpau_ctx *ctx);
hw_vdpau_ctx hw_vdpau_ctx_copy(hw_vdpau_ctx *ctx);

void hw_vdpau_frame_init(hw_vdpau_frame *frame);
void hw_vdpau_frame_unref(hw_vdpau_frame *frame);
hw_vdpau_frame hw_vdpau_frame_copy(const hw_vdpau_frame *frame);


#endif
