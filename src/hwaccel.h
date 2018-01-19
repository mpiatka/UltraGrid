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

struct hw_vdpau_frame_data{
        VdpVideoSurface *surface;
};


#endif
