#include <juice/juice.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <poll.h>
#include <arpa/inet.h>
#include <netdb.h>
#include "utils/udp_holepunch.h"

#include "debug.h"

#ifdef _WIN32
#include <windows.h>
#include <winsock.h>
static void sleep(unsigned int secs) { Sleep(secs * 1000); }
#else
#include <unistd.h>
#endif

#define MAX_MSG_LEN 2048
#define MSG_HEADER_LEN 5
#define MOD_NAME "[HOLEPUNCH] "

struct Punch_ctx {
        juice_agent_t *juice_agent;

        int coord_sock;
        int local_candidate_port;
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
        UNUSED(agent);
        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Received candidate: %s\n", sdp);
        struct Punch_ctx *ctx = (struct Punch_ctx *) user_ptr;
        send_msg(ctx->coord_sock, sdp);

        /* sdp is a RFC5245 string which should look like this:
         * "a=candidate:2 1 UDP <prio> <ip> <port> typ <type> ..."
         * Since libjuice reports only the external (after NAT translation)
         * receive port, we need to get the receive port number from the local
         * candidate (of type "host").
         */

        const char *c = sdp;
        //port is located after 5 space characters
        for(int i = 0; i < 5; i++){
                c = strchr(c, ' ') + 1;
                assert(c);
        }
        char *end;
        int port = strtol(c, &end, 10);
        assert(c != end);
        assert(*end == ' ');
        c = end + 1;

        const char *host_type_str = "typ host";
        if(strncmp(c, host_type_str, strlen(host_type_str)) == 0){
                log_msg(LOG_LEVEL_INFO, MOD_NAME "Local candidate port: %d\n", port);
                ctx->local_candidate_port = port;
        }
}

static juice_agent_t *create_agent(const struct Holepunch_config *c, void *usr_ptr){
        juice_config_t conf;
        memset(&conf, 0, sizeof(conf));

        conf.stun_server_host = c->stun_srv_addr;
        conf.stun_server_port = c->stun_srv_port;

        conf.turn_servers = NULL;
        conf.turn_servers_count = 0;

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
        if(!host){
                error_msg(MOD_NAME "Failed to resolve coordination server host\n");
                return false;
        }

        memcpy(&sockaddr.sin_addr.s_addr, host->h_addr_list[0], host->h_length);
        if (connect(s, (struct sockaddr *) &sockaddr, sizeof(sockaddr)) < 0){
                error_msg(MOD_NAME "Failed to connect to coordination server\n");
                return false;
        }

        *sock = s;
        return true;
}

static void exchange_coord_desc(juice_agent_t *agent, int coord_sock){
        char sdp[JUICE_MAX_SDP_STRING_LEN];
        juice_get_local_description(agent, sdp, JUICE_MAX_SDP_STRING_LEN);
        log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Local description:\n%s\n", sdp);

        send_msg(coord_sock, sdp);

        char msg_buf[MAX_MSG_LEN];
        recv_msg(coord_sock, msg_buf, sizeof(msg_buf));
        log_msg(LOG_LEVEL_INFO, MOD_NAME "Remote client name: %s\n", msg_buf);
        recv_msg(coord_sock, msg_buf, sizeof(msg_buf));
        log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Remote desc: %s\n", msg_buf);

        juice_set_remote_description(agent, msg_buf);
}

static void discover_and_xchg_candidates(juice_agent_t *agent, int coord_sock) {
        juice_gather_candidates(agent);

        struct pollfd fds;
        fds.fd = coord_sock;
        fds.events = POLLIN;

        while(1){
                poll(&fds, 1, 300);
                if(fds.revents & POLLIN){
                        char msg_buf[MAX_MSG_LEN];
                        recv_msg(coord_sock, msg_buf, sizeof(msg_buf));
                        log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Received remote candidate\n");
                        juice_add_remote_candidate(agent, msg_buf);
                }
                juice_state_t state = juice_get_state(agent);
                if(state == JUICE_STATE_COMPLETED)
                        break;
        }
}

static bool initialize_punch(struct Punch_ctx *ctx,
                const struct Holepunch_config *c,
                const char *room_suffix)
{
        char room_name[512] = "";
        size_t room_len = strlen(c->room_name);
        if(room_len + strlen(room_suffix) + 1 > sizeof(room_name)){
                error_msg(MOD_NAME "Room name too long\n");
                return false;
        }
        strncat(room_name, c->room_name, sizeof(room_name) - 1);
        strncat(room_name, room_suffix, sizeof(room_name) - room_len - 1);

        if(!connect_to_coordinator(c->coord_srv_addr, c->coord_srv_port, &ctx->coord_sock)){
                return false;
        }

        send_msg(ctx->coord_sock, c->client_name);
        send_msg(ctx->coord_sock, room_name);

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

static void juice_log_handler(juice_log_level_t level, const char *message){
        UNUSED(level);
        log_msg(LOG_LEVEL_DEBUG, MOD_NAME "libjuice: %s\n", message);
}

static bool run_punch(struct Punch_ctx *ctx, char *local, char *remote){
        discover_and_xchg_candidates(ctx->juice_agent, ctx->coord_sock);

        enum juice_state state = juice_get_state(ctx->juice_agent);
        if(state != JUICE_STATE_COMPLETED){
                error_msg(MOD_NAME "Punching finished unsuccessfuly (state %d)\n", state);
                return false;
        }

        if ((juice_get_selected_addresses(ctx->juice_agent,
                                        local,
                                        JUICE_MAX_CANDIDATE_SDP_STRING_LEN,
                                        remote,
                                        JUICE_MAX_CANDIDATE_SDP_STRING_LEN) != 0))
        {
                error_msg(MOD_NAME, "Failed to read selected addresses\n");
                return false;
        }

        log_msg(LOG_LEVEL_INFO, MOD_NAME "Local candidate  : %s\n", local);
        log_msg(LOG_LEVEL_INFO, MOD_NAME "Remote candidate : %s\n", remote);

        return true;
}

static void cleanup_punch(struct Punch_ctx *ctx){
        juice_destroy(ctx->juice_agent);
        close(ctx->coord_sock);

}

bool punch_udp(struct Holepunch_config c){
        struct Punch_ctx video_ctx = {0};
        struct Punch_ctx audio_ctx = {0};

        juice_set_log_level(JUICE_LOG_LEVEL_DEBUG);
        juice_set_log_handler(juice_log_handler);

        char local[JUICE_MAX_CANDIDATE_SDP_STRING_LEN];
        char remote[JUICE_MAX_CANDIDATE_SDP_STRING_LEN];

        char name[512] = {};
        if(!c.client_name){
                gethostname(name, sizeof(name) - 1);
                c.client_name = name;
        }

        if(!initialize_punch(&video_ctx, &c, "_video")){
                return false;
        }

        if(!run_punch(&video_ctx, local, remote))
                return false;

        *c.video_rx_port = video_ctx.local_candidate_port;
        assert(split_host_port(remote, c.video_tx_port));

        strncpy(c.host_addr, remote, c.host_addr_len);

        if(!initialize_punch(&audio_ctx, &c, "_audio")){
                cleanup_punch(&video_ctx);
                return false;
        }

        cleanup_punch(&video_ctx);

        if(!run_punch(&audio_ctx, local, remote))
                return false;

        *c.audio_rx_port = audio_ctx.local_candidate_port;
        assert(split_host_port(remote, c.audio_tx_port));

        assert(strcmp(c.host_addr, remote) == 0);

        log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Cleaning up\n");
        cleanup_punch(&audio_ctx);

        log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Cleanup done\n");


        return true;
}
