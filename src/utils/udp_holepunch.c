#include <juice/juice.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <arpa/inet.h>
#include <netdb.h>
#include "utils/udp_holepunch.h"

#ifdef _WIN32
#include <windows.h>
static void sleep(unsigned int secs) { Sleep(secs * 1000); }
#else
#include <unistd.h> // for sleep
#endif

#define MAX_MSG_LEN 2048
#define MSG_HEADER_LEN 5

struct Punch_ctx {
        juice_agent_t *juice_agent;

        int coord_sock;
};

static void send_msg(int sock, const char *msg){
        size_t msg_size = strlen(msg);
        assert(msg_size < MAX_MSG_LEN);

        char header[MSG_HEADER_LEN];
        memset(header, ' ', sizeof(header));
        snprintf(header, sizeof(header), "%lu", msg_size);

        send(sock, header, sizeof(header), 0);

        send(sock, msg, msg_size, 0);
}

static void on_candidate(juice_agent_t *agent, const char *sdp, void *user_ptr) {
        printf("Candidate: %s\n", sdp);
        struct Punch_ctx *ctx = (struct Punch_ctx *) user_ptr;
        send_msg(ctx->coord_sock, sdp);
}

static juice_agent_t *create_agent(const struct Holepunch_config *c, void *usr_ptr){
        juice_config_t conf;
        memset(&conf, 0, sizeof(conf));

        conf.stun_server_host = c->stun_srv_addr;
        conf.stun_server_port = c->stun_srv_port;

        conf.turn_servers = NULL;
        conf.turn_servers_count = 0;

#if 0
        conf.cb_state_changed = on_state_changed1;
        conf.cb_gathering_done = on_gathering_done1;
        conf.cb_recv = on_recv1;
#endif
        conf.cb_candidate = on_candidate;
        conf.user_ptr = usr_ptr;

        return juice_create(&conf);
}


static size_t recv_msg(int sock, char *buf, size_t buf_len){
        char header[MSG_HEADER_LEN + 1];

        int bytes = recv(sock, header, MSG_HEADER_LEN, MSG_WAITALL);
        if(bytes != MSG_HEADER_LEN){
                return 0;
        }
        header[MSG_HEADER_LEN] = '\0';

        unsigned expected_len;
        char *end;
        expected_len = strtol(header, &end, 10);
        if(header == end){
                return 0;
        }

        if(expected_len > buf_len)
                expected_len = buf_len;

        bytes = recv(sock, buf, expected_len, MSG_WAITALL);
        buf[bytes] = '\0';

        return bytes;
}

static bool connect_to_coordinator(const char *coord_srv_addr,
                int coord_srv_port,
                int *sock)
{
        int s = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);

        struct sockaddr_in sockaddr;
        memset(&sockaddr, 0, sizeof(sockaddr));
        sockaddr.sin_family = AF_INET;
        sockaddr.sin_port = htons(coord_srv_port);
        struct hostent *host = gethostbyname(coord_srv_addr);
        memcpy(&sockaddr.sin_addr.s_addr, host->h_addr_list[0], host->h_length);

        if (connect(s, (struct sockaddr *) &sockaddr, sizeof(sockaddr)) < 0){
                return false;
        }

        *sock = s;
        return true;
}

static bool exchange_coord_desc(juice_agent_t *agent, int coord_sock){
        char sdp[JUICE_MAX_SDP_STRING_LEN];
        juice_get_local_description(agent, sdp, JUICE_MAX_SDP_STRING_LEN);
        printf("Local description:\n%s\n", sdp);

        send_msg(coord_sock, sdp);

        char msg_buf[MAX_MSG_LEN];
        recv_msg(coord_sock, msg_buf, sizeof(msg_buf));
        printf("Remote client name: %s\n", msg_buf);
        recv_msg(coord_sock, msg_buf, sizeof(msg_buf));
        printf("Remote desc: %s\n", msg_buf);

        juice_set_remote_description(agent, msg_buf);
}

static bool discover_and_xchg_candidates(juice_agent_t *agent, int coord_sock,
                char *local, char *remote)
{
        juice_gather_candidates(agent);

        fd_set rfds;
        FD_ZERO(&rfds);

        struct timeval tv;
        tv.tv_sec = 1;
        tv.tv_usec = 0;

        while(1){
                FD_SET(coord_sock, &rfds);
                int res = select(coord_sock + 1, &rfds, NULL, NULL, &tv);
                if(FD_ISSET(coord_sock, &rfds)){
                        char msg_buf[MAX_MSG_LEN];
                        recv_msg(coord_sock, msg_buf, sizeof(msg_buf));
                        printf("Received remote candidate\n");
                        juice_add_remote_candidate(agent, msg_buf);
                }
                juice_state_t state = juice_get_state(agent);
                if(state == JUICE_STATE_COMPLETED)
                        break;
        }

        if ((juice_get_selected_addresses(agent,
                                        local,
                                        JUICE_MAX_CANDIDATE_SDP_STRING_LEN,
                                        remote,
                                        JUICE_MAX_CANDIDATE_SDP_STRING_LEN) == 0)) {
                printf("Local candidate  1: %s\n", local);
                printf("Remote candidate 1: %s\n", remote);
        }
}

static bool initialize_punch(struct Punch_ctx *ctx, const struct Holepunch_config *c){
        if(!connect_to_coordinator(c->coord_srv_addr, c->coord_srv_port, &ctx->coord_sock)){
                fprintf(stderr, "Failed to connect to coordinator!\n");
                return false;
        }

        send_msg(ctx->coord_sock, c->client_name);
        send_msg(ctx->coord_sock, c->room_name);

        ctx->juice_agent = create_agent(c, ctx);
        
        exchange_coord_desc(ctx->juice_agent, ctx->coord_sock);

        return true;
}

static bool split_host_port(char *pair, int *port){
        char *colon = strrchr(pair, ':');
        if(!colon)
                return false;

        *colon = '\0';

        char *end;
        int p = strtol(colon + 1, &end, 10);
        if(end == colon + 1)
                return false;

        *port = p;
        return true;
}

bool punch_udp(const struct Holepunch_config *c){
        struct Punch_ctx video_ctx = {0};

        juice_set_log_level(JUICE_LOG_LEVEL_DEBUG);

        if(!initialize_punch(&video_ctx, c)){
                return false;
        }

        char local[JUICE_MAX_CANDIDATE_SDP_STRING_LEN];
        char remote[JUICE_MAX_CANDIDATE_SDP_STRING_LEN];
        discover_and_xchg_candidates(video_ctx.juice_agent,
                        video_ctx.coord_sock,
                        local, remote);

        juice_destroy(video_ctx.juice_agent);
        close(video_ctx.coord_sock);

        assert(split_host_port(local, c->video_rx_port));
        assert(split_host_port(remote, c->video_tx_port));

        strncpy(c->host_addr, remote, c->host_addr_len);

        return true;
}
