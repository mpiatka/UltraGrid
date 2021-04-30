#ifndef UG_UTILS_UDP_HOLEPUNCH_H
#define UG_UTILS_UDP_HOLEPUNCH_H

#ifndef __cplusplus
#include <stdbool.h>
#endif // ! defined __cplusplus

#ifdef __cplusplus
extern "C" {
#endif

struct Holepunch_config{
        const char *client_name;
        const char *room_name;

        int *video_rx_port;
        int *video_tx_port;
        int *audio_rx_port;
        int *audio_tx_port;

        char *host_addr;
        size_t host_addr_len;

        const char *coord_srv_addr;
        int coord_srv_port;
        const char *stun_srv_addr;
        int stun_srv_port;
};

bool punch_udp(const struct Holepunch_config *c);


#ifdef __cplusplus
} //extern "C"
#endif

#endif
