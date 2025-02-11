#ifndef LIBULTRAGRID_H_101e9ad725bd
#define LIBULTRAGRID_H_101e9ad725bd

#ifdef __cplusplus
extern "C" {
#endif

#define LIBULTRAGRID_HEADER_VERSION 3

unsigned libug_get_version();

enum libug_mode{
        LIBUG_MODE_NONE = 0,
        LIBUG_MODE_SEND = 1,
        LIBUG_MODE_RECV = 1 << 1,
};

typedef void (*libug_frame_recv_fn)(void *user, struct video_frame *f);

typedef struct libug_conf{
        enum libug_mode mode;
        void *user_ptr;

        //Send params
        const char *send_destination;
        const char *compress;
        const char *fec;
        int tx_port;
        int mtu;
        long long rate_limit;

        //Recv params
        int rx_port;
        libug_frame_recv_fn frame_callback;
} libug_conf;

void libug_init_conf(libug_conf *conf);


struct libug_handle;

struct libug_handle *libug_create_handle(const struct libug_conf *conf);
void libug_destroy_handle(struct libug_handle *handle);

void libug_start_recv(struct libug_handle *h);

void libug_send_video_frame(struct libug_handle *h, struct video_frame *frame);

void libug_set_log_callback(void(*log_callback)(const char *));
#ifdef __cplusplus
}
#endif

#endif
