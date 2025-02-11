#include "libultragrid.h"
#include <thread>
#include <stdio.h>
#include "video_rxtx.hpp"                         // for param_u, video_rxtx
#include "video_frame.h"                         // for param_u, video_rxtx
#include "audio/types.h"
#include "video_codec.h"
#include "video_rxtx/ultragrid_rtp.hpp"
#include "video_display.h"
#include "video_display/pipe.hpp"
#include "debug.h"

unsigned libug_get_version(){
	return LIBULTRAGRID_HEADER_VERSION;
}

struct libug_handle : public frame_recv_delegate{
        virtual ~libug_handle() {  }
        struct module root_module;
        video_rxtx* video_rxtx_state{};
        struct vidcap *capture_device{};
        struct display *display{};

        virtual void frame_arrived(struct video_frame *, struct audio_frame *) override;
        libug_frame_recv_fn frame_recieved{};
        void *user = nullptr;
        std::string compress_str;
        std::string fec_str;
        std::string destination_str;

        std::thread recv_thread;
};

void libug_handle::frame_arrived(struct video_frame *v, struct audio_frame *a){
        if(a){
                AUDIO_FRAME_DISPOSE(a);
        }

        if(v){
                if(frame_recieved){
                        frame_recieved(user, v);
                }

                VIDEO_FRAME_DISPOSE(v);
        }
}

void libug_init_conf(libug_conf *conf){
        *conf = {};

        conf->send_destination = "localhost";
        conf->compress = "none";
        conf->fec = "none";
        conf->rx_port = 7004;
        conf->tx_port = 5004;

        conf->mtu = 1500;
        conf->rate_limit = RATE_DYNAMIC;
}

struct libug_handle *libug_create_handle(const struct libug_conf *conf){
        library_preinit();
        auto handle = std::make_unique<libug_handle>();

        handle->frame_recieved = conf->frame_callback;
        handle->user = conf->user_ptr;

        struct common_opts common = { COMMON_OPTS_INIT };
        init_root_module(&handle->root_module);
        common.parent = &handle->root_module;
        common.mtu = conf->mtu;


        if(conf->compress)
                handle->compress_str = conf->compress;
        if(conf->send_destination)
                handle->destination_str = conf->send_destination;
        if(conf->fec)
                handle->fec_str = conf->fec;

        int rx_port = 0;
        if(conf->mode & LIBUG_MODE_RECV){
                rx_port = conf->rx_port;
        }

        std::map<std::string, param_u> params;
        // common
        params["compression"].str = handle->compress_str.c_str();
        params["rxtx_mode"].i = MODE_SENDER; //TODO

        //RTP
        params["common"].cptr = static_cast<const void *>(&common);
        params["receiver"].str = handle->destination_str.c_str();
        params["rx_port"].i = rx_port;
        params["tx_port"].i = conf->tx_port;
        params["fec"].str = handle->fec_str.c_str();
        params["bitrate"].ll = conf->rate_limit;

        printf("Rx_port=%d\n", rx_port);

        params["decoder_mode"].l = VIDEO_NORMAL;

        if(conf->mode & LIBUG_MODE_RECV){
                char cfg[128] = "";
                int ret;

                snprintf(cfg, sizeof cfg, "%p:codec=RGBA", handle.get());
                ret = initialize_video_display(&handle->root_module, "pipe", cfg, 0, NULL, &handle->display);
		if(!handle->display){
			printf("Failed to init display");
			return nullptr;
		}

                // UltraGrid RTP
        }
        params["display_device"].ptr = handle->display;

        handle->video_rxtx_state = video_rxtx::create("ultragrid_rtp", params);

        return handle.release();
}


void libug_destroy_handle(struct libug_handle *handle){
        if(!handle)
                return;

        root_module_exit(&handle->root_module, 0);

        printf("destroyUg begin @ %p\n", handle);
        handle->video_rxtx_state->join();
        printf("joined rxtx\n");
        if(handle->recv_thread.joinable()){
                handle->recv_thread.join();
                printf("joined recv thread\n");
        }
        delete handle->video_rxtx_state;
        if(handle->display){
                display_put_frame(handle->display, nullptr, PUTF_DISCARD);
                printf("poisoned display\n");
                display_join(handle->display);
                printf("joined display\n");
        }
        delete handle;
        printf("destroyUg end @ %p\n", handle);
}

void libug_send_video_frame(libug_handle *h, struct video_frame *frame){
        std::shared_ptr<video_frame> f(frame, vf_free);
        h->video_rxtx_state->send(std::move(f));
}

void libug_start_recv(libug_handle *h){
        h->recv_thread = std::thread(video_rxtx::receiver_thread, h->video_rxtx_state);
        display_run_new_thread(h->display);
}

void libug_set_log_callback(void(*log_callback)(const char *)){
	get_log_output().set_log_callback(log_callback);
}
