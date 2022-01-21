/**
 * @file   hd-rum-translator/hd-rum-recompress.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * Component of the transcoding reflector that takes an uncompressed frame,
 * recompresses it to another compression and sends it to destination
 * (therefore it wraps the whole sending part of UltraGrid).
 */
/*
 * Copyright (c) 2013-2019 CESNET, z. s. p. o.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of CESNET nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <cinttypes>
#include <chrono>
#include <memory>
#include <thread>


#include "hd-rum-translator/hd-rum-recompress.h"

#include "debug.h"
#include "host.h"
#include "rtp/rtp.h"

#include "video_compress.h"

#include "video_rxtx/ultragrid_rtp.h"

namespace {
struct compress_state_deleter{
        void operator()(struct compress_state *s){ module_done(CAST_MODULE(s)); }
};
}

using namespace std;

struct recompress_output_port {
        recompress_output_port(struct module *parent,
                std::string host, unsigned short rx_port,
                unsigned short tx_port, int mtu, const char *fec, long long bitrate);

        std::unique_ptr<ultragrid_rtp_video_rxtx> video_rxtx;
        std::string host;
        int rx_port;
        int tx_port;
        int mtu;
        std::string fec;
        long long bitrate;

        std::chrono::steady_clock::time_point t0;
        int frames;

        bool active;
};

struct recompress_worker_ctx {
        std::string compress_cfg;
        std::unique_ptr<compress_state, compress_state_deleter> compress;

        std::mutex ports_mut;
        std::vector<recompress_output_port> ports;

        std::thread thread;
};

struct state_recompress {
        struct module *parent;
        std::map<std::string, recompress_worker_ctx> workers;
        std::vector<std::pair<std::string, int>> index_to_port;
};

recompress_output_port::recompress_output_port(struct module *parent,
                std::string host, unsigned short rx_port,
                unsigned short tx_port, int mtu, const char *fec, long long bitrate) :
        host(std::move(host)),
        rx_port(rx_port),
        tx_port(tx_port),
        mtu(mtu),
        fec(fec ? fec : ""),
        bitrate(bitrate),
        frames(0),
        active(true)
{
        int force_ip_version = 0;
        auto start_time = std::chrono::steady_clock::now();

        std::map<std::string, param_u> params;

        // common
        params["parent"].ptr = parent;
        params["exporter"].ptr = NULL;
        params["compression"].str = "none";
        params["rxtx_mode"].i = MODE_SENDER;
        params["paused"].b = false;

        //RTP
        params["mtu"].i = mtu;
        params["receiver"].str = this->host.c_str();
        params["rx_port"].i = rx_port;
        params["tx_port"].i = tx_port;
        params["force_ip_version"].i = force_ip_version;
        params["mcast_if"].str = NULL;
        params["fec"].str = fec;
        params["encryption"].str = NULL;
        params["bitrate"].ll = bitrate;
        params["start_time"].cptr = (const void *) &start_time;
        params["video_delay"].vptr = 0;

        // UltraGrid RTP
        params["decoder_mode"].l = VIDEO_NORMAL;
        params["display_device"].ptr = NULL;

        auto rxtx = video_rxtx::create("ultragrid_rtp", params);
        if (host.find(':') != std::string::npos) {
                rxtx->m_port_id = "[" + host + "]:" + to_string(tx_port);
        } else {
                rxtx->m_port_id = host + ":" + to_string(tx_port);
        }

        video_rxtx.reset(dynamic_cast<ultragrid_rtp_video_rxtx *>(rxtx));
}

static void recompress_port_write(recompress_output_port& port, shared_ptr<video_frame> frame)
{
        port.frames += 1;

        auto now = chrono::steady_clock::now();

        double seconds = chrono::duration_cast<chrono::seconds>(now - port.t0).count();
        if(seconds > 5) {
                double fps = port.frames / seconds;
                log_msg(LOG_LEVEL_INFO, "[0x%08" PRIx32 "->%s:%d:0x%08" PRIx32 "] %d frames in %g seconds = %g FPS\n",
                                frame->ssrc,
                                port.host.c_str(), port.tx_port,
                                port.video_rxtx->get_ssrc(),
                                port.frames, seconds, fps);
                port.t0 = now;
                port.frames = 0;
        }

        port.video_rxtx->send(frame);
}

static void recompress_worker(struct recompress_worker_ctx *ctx){
        assert(ctx->compress);

        while(auto frame = compress_pop(ctx->compress.get())){
                if(!frame){
                        //poisoned
                        break;
                }

                std::lock_guard<std::mutex>(ctx->ports_mut);
                for(auto& port : ctx->ports){
                        recompress_port_write(port, frame);
                }
        }
}

int recompress_add_port(struct state_recompress *s,
		const char *host, const char *compress, unsigned short rx_port,
		unsigned short tx_port, int mtu, const char *fec, long long bitrate)
{
        auto port = recompress_output_port(s->parent, host, rx_port, tx_port,
                        mtu, fec, bitrate);

        auto& worker = s->workers[compress];
        if(!worker.compress){
                worker.compress_cfg = compress;
                compress_state *cmp = nullptr;
                //TODO error check
                int ret = compress_init(s->parent, compress, &cmp);
                worker.compress.reset(cmp);

                worker.thread = std::thread(recompress_worker, &worker);
        }

        int index_in_worker = -1;
        {
                std::lock_guard<std::mutex>(worker.ports_mut);
                index_in_worker = worker.ports.size();
                worker.ports.push_back(std::move(port));
        }

        int index_of_port = s->index_to_port.size();
        s->index_to_port.emplace_back(compress, index_in_worker);

        return index_of_port;
}

void recompress_remove_port(struct state_recompress *s, int index){
        auto [compress_cfg, i] = s->index_to_port[index];

        auto& worker = s->workers[compress_cfg];
        {
                std::unique_lock<std::mutex> lock(worker.ports_mut);
                worker.ports[i].video_rxtx->join();
                worker.ports.erase(worker.ports.begin() + i);
        }
        s->index_to_port.erase(s->index_to_port.begin() + index);

        if(worker.ports.empty()){
                //poison compress
                compress_frame(worker.compress.get(), nullptr);
                worker.thread.join();
                s->workers.erase(compress_cfg);
        }
}

uint32_t recompress_get_port_ssrc(struct state_recompress *s, int idx){
        auto [compress_cfg, i] = s->index_to_port[idx];

        return s->workers[compress_cfg].ports[i].video_rxtx->get_ssrc();
}

void recompress_port_set_active(struct state_recompress *s,
                int index, bool active)
{
        auto [compress_cfg, i] = s->index_to_port[index];

        std::unique_lock<std::mutex>(s->workers[compress_cfg].ports_mut);
        s->workers[compress_cfg].ports[i].active = active;
}

int recompress_get_num_active_ports(struct state_recompress *s){
        int ret = 0;
        for(const auto& worker : s->workers){
                for(const auto& port : worker.second.ports){
                        if(port.active)
                                ret++;
                }
        }

        return ret;
}

struct state_recompress *recompress_init(struct module *parent) {
        auto state = new state_recompress();
        if(!state)
                return nullptr;

        state->parent = parent;

        return state;
}

void recompress_process_async(state_recompress *s, std::shared_ptr<video_frame> frame){
        for(const auto& worker : s->workers){
                compress_frame(worker.second.compress.get(), frame);
        }
}

void recompress_done(struct state_recompress *s) {
        for(auto& worker : s->workers){
                //poison compress
                compress_frame(worker.second.compress.get(), nullptr);

                for(const auto& port : worker.second.ports){
                        port.video_rxtx->join();
                }
                worker.second.thread.join();
        }
        delete s;
}

