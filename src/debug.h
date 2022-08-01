/*
 * FILE:    debug.h
 * PROGRAM: RAT
 * AUTHOR:  Isidor Kouvelas + Colin Perkins + Mark Handley + Orion Hodson
 * 
 * $Revision: 1.1 $
 * $Date: 2007/11/08 09:48:59 $
 *
 * Copyright (c) 1995-2000 University College London
 * Copyright (c) 2005-2021 CESNET, z. s. p. o.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions 
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *      This product includes software developed by the Computer Science
 *      Department at University College London
 * 4. Neither the name of the University nor of the Department may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#ifndef _RAT_DEBUG_H
#define _RAT_DEBUG_H

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdbool.h>
#include <stdint.h>
#endif // defined __cplusplus

#define UNUSED(x)	(x=x)

#define LOG_LEVEL_QUIET   0 ///< suppress all logging
#define LOG_LEVEL_FATAL   1 ///< errors that prevent UG run
#define LOG_LEVEL_ERROR   2 ///< general errors
#define LOG_LEVEL_WARNING 3 ///< less severe errors
#define LOG_LEVEL_NOTICE  4 ///< information that may be interesting
#define LOG_LEVEL_INFO    5 ///< normal reporting
#define LOG_LEVEL_VERBOSE 6 ///< display more messages but no more than
                            ///< one of a kind every ~1 sec
#define LOG_LEVEL_DEBUG   7 ///< like LOG_LEVEL_VERBOSE, freq. approx.
                            ///< 1 message every video frame
#define LOG_LEVEL_DEBUG2  8 ///< even more verbose - eg. every packet can
                            ///< be logged
#define LOG_LEVEL_MAX LOG_LEVEL_DEBUG2
extern volatile int log_level;

enum log_timestamp_mode{
	LOG_TIMESTAMP_DISABLED = 0,
	LOG_TIMESTAMP_ENABLED = 1,
	LOG_TIMESTAMP_AUTO = -1
};

#ifdef __cplusplus
extern "C" {
#endif

void debug_dump(void*lp, int len);
#ifdef DEBUG
void debug_file_dump(const char *key, void (*serialize)(const void *data, FILE *), void *data);
#else
#define debug_file_dump(key, serialize, data) (void) (key), (void) (serialize), (void) (data)
#endif

#ifndef ATTRIBUTE
#define ATTRIBUTE(a)
#endif

#define error_msg(...) log_msg(LOG_LEVEL_ERROR, __VA_ARGS__)
#define verbose_msg(...) log_msg(LOG_LEVEL_VERBOSE, __VA_ARGS__)
///#define debug_msg(...) log_msg(LOG_LEVEL_DEBUG, "[pid/%d +%d %s] ", getpid(), __LINE__, __FILE__), log_msg(LOG_LEVEL_DEBUG, __VA_ARGS__)
#define debug_msg(...) log_msg(LOG_LEVEL_DEBUG, __VA_ARGS__)
void log_msg(int log_level, const char *format, ...) ATTRIBUTE(format (printf, 2, 3));
void log_msg_once(int log_level, uint32_t id, const char *msg);
void log_perror(int log_level, const char *msg);

bool parse_log_cfg(const char *conf_str,
		int *log_lvl,
		bool *logger_skip_repeats,
		enum log_timestamp_mode *show_timestamps);

#ifdef __cplusplus
}
#endif

#define CUMULATIVE_REPORTS_INTERVAL 30

#ifdef __cplusplus
#include <atomic>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <mutex>
#include "compat/platform_time.h"
#include "rang.hpp"

class keyboard_control; // friend

class Log_output{
public:
        Log_output() = default;
        
        std::string& get_buffer() { return buffer; }
        void submit();

        void set_skip_repeats(bool val) { skip_repeated.store(val, std::memory_order_relaxed); }

        void set_timestamp_mode(log_timestamp_mode val) { show_timestamps = val; }

        Log_output(const Log_output&) = delete;
        Log_output(Log_output&&) = delete;

        Log_output& operator=(const Log_output&) = delete;
        Log_output& operator=(Log_output&&) = delete;

private:

        thread_local static std::string buffer;

        std::atomic<bool> skip_repeated;
        log_timestamp_mode show_timestamps;


        /* Since writing to stdout uses locks internally anyway (C11 standard
         * 7.21.2 sections 7&8), using a mutex here does not cause any significant
         * overhead, we just wait for the lock a bit earlier. */
        std::mutex mut;
        std::string last_msg;
        int last_msg_repeats = 0;

        friend class keyboard_control;
};

inline void Log_output::submit(){
        std::ostringstream timestamp;
        if (show_timestamps == 1 || (show_timestamps == -1 && log_level >= LOG_LEVEL_VERBOSE)) {
                auto time_ms = time_since_epoch_in_ms();
                timestamp << "[" << std::fixed << std::setprecision(3) << time_ms / 1000.0  << "] ";
        }

        std::lock_guard<std::mutex> lock(mut);
        if (skip_repeated && rang::rang_implementation::isTerminal(std::clog.rdbuf())) {
                if (buffer == last_msg) {
                        last_msg_repeats++;
                        std::clog << "    Last message repeated " << last_msg_repeats << " times\r";
                        std::clog.flush();
                        return;
                }

                if (last_msg_repeats > 0) {
                        std::clog << "\n";
                }
                last_msg_repeats = 0;
        }


        std::clog << timestamp.str() << buffer << rang::style::reset << rang::fg::reset;

        std::swap(last_msg, buffer);
}

inline Log_output& get_log_output(){
        static Log_output out;
        return out;
}

// Log, version 0.1: a simple logging class
class Logger
{
public:
        static void preinit();
        inline Logger(int l) : level(l) {
                rang::fg color = rang::fg::reset;
                rang::style style = rang::style::reset;

                switch (level) {
                case LOG_LEVEL_FATAL:   color = rang::fg::red; style = rang::style::bold; break;
                case LOG_LEVEL_ERROR:   color = rang::fg::red; break;
                case LOG_LEVEL_WARNING: color = rang::fg::yellow; break;
                case LOG_LEVEL_NOTICE:  color = rang::fg::green; break;
                }

                oss << style << color;
        }

        inline ~Logger() {
                std::string msg = oss.str();

                auto& buf = get_log_output().get_buffer();
                buf.clear();

                buf += msg;

                get_log_output().submit();

        }
        inline std::ostream& Get() {
                return oss;
        }
        inline void once(uint32_t id, const std::string &msg) {
                if (oneshot_messages.count(id) > 0) {
                        return;
                }
                oneshot_messages.insert(id);
                oss << msg;
        }

private:
        int level;
        std::ostringstream oss;

        static thread_local std::set<uint32_t> oneshot_messages;
};

#define LOG(level) \
if (level <= log_level) Logger(level).Get()

#define LOG_ONCE(level, id, msg) \
if (level <= log_level) Logger(level).once(id, msg)

#endif

#ifdef DEBUG
#define DEBUG_TIMER_EVENT(name) struct timeval name = { 0, 0 }; gettimeofday(&name, NULL)
#define DEBUG_TIMER_START(name) DEBUG_TIMER_EVENT(name##_start);
#define DEBUG_TIMER_STOP(name) DEBUG_TIMER_EVENT(name##_stop); log_msg(LOG_LEVEL_DEBUG2, "%s duration: %lf s\n", #name, tv_diff(name##_stop, name##_start)) // NOLINT(cppcoreguidelines-pro-type-vararg, hicpp-vararg)
#else
#define DEBUG_TIMER_START(name)
#define DEBUG_TIMER_STOP(name)
#endif

#endif
