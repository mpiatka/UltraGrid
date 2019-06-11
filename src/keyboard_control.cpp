/**
 * @file   keyboard_control.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * With code taken from Olivier Mehani from http://shtrom.ssji.net/skb/getc.html
 */
/*
 * Copyright (c) 2015 CESNET, z. s. p. o.
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
#endif // HAVE_CONFIG_H

#include "debug.h"
#include "host.h"
#include "keyboard_control.h"
#include "messaging.h"
#include "video.h"

#include <iomanip>
#include <iostream>
#include <sstream>

#ifdef HAVE_TERMIOS_H
#include <unistd.h>
#include <termios.h>
#else
#include <conio.h>
#endif

#define CTRL_X 24

using namespace std;

#ifdef HAVE_TERMIOS_H
static bool signal_catched = false;

static void catch_signal(int)
{
        signal_catched = true;
}
#endif

#ifdef HAVE_TERMIOS_H
static struct termios old_tio;
#endif

#ifdef HAVE_TERMIOS_H
static void restore_old_tio(void)
{
        struct sigaction sa, sa_old;
        memset(&sa, 0, sizeof sa);
        sa.sa_handler = SIG_IGN;
        sigemptyset(&sa.sa_mask);
        sigaction(SIGTTOU, &sa, &sa_old);
        /* set the new settings immediately */
        /* restore the former settings */
        tcsetattr(STDIN_FILENO,TCSANOW,&old_tio);
        sigaction(SIGTTOU, &sa_old, NULL);
}
#endif

keyboard_control::keyboard_control() :
        m_root(nullptr),
#ifdef HAVE_TERMIOS_H
        m_should_exit_pipe{0, 0},
#else
        m_should_exit(false),
#endif
        m_started(false),
        m_locked_against_changes(true)
{
        m_start_time = time(NULL);
}

ADD_TO_PARAM(disable_keyboard_control, "disable-keyboard-control", "* disable-keyboard-control\n"
                "  disables keyboard control (usable mainly for non-interactive runs)\n");
void keyboard_control::start(struct module *root)
{
        if (get_commandline_param("disable-keyboard-control")) {
                return;
        }
        m_root = root;
#ifdef HAVE_TERMIOS_H
        if (pipe(m_should_exit_pipe) != 0) {
                log_msg(LOG_LEVEL_ERROR, "[key control] Cannot create control pipe!\n");
                return;
        }

        if (!isatty(STDIN_FILENO)) {
                log_msg(LOG_LEVEL_WARNING, "[key control] Stdin is not a TTY - disabling keyboard control.\n");
                return;
        }

        struct termios new_tio;
        /* get the terminal settings for stdin */
        tcgetattr(STDIN_FILENO,&old_tio);
        /* we want to keep the old setting to restore them a the end */
        new_tio=old_tio;
        /* disable canonical mode (buffered i/o) and local echo */
        new_tio.c_lflag &=(~ICANON & ~ECHO);
        // Wrap calling of tcsetattr() by handling SIGTTOU. SIGTTOU can be raised if task is
        // run in background and trying to call tcsetattr(). If so, we disable keyboard
        // control.
        struct sigaction sa, sa_old;
        memset(&sa, 0, sizeof sa);
        sa.sa_handler = catch_signal;
        sigemptyset(&sa.sa_mask);
        sigaction(SIGTTOU, &sa, &sa_old);
        /* set the new settings immediately */
        tcsetattr(STDIN_FILENO,TCSANOW,&new_tio);
        sigaction(SIGTTOU, &sa_old, NULL);
        if (signal_catched) {
                log_msg(LOG_LEVEL_WARNING, "[key control] Background task - disabling keyboard control.\n");
                return;
        }
        atexit(restore_old_tio);
#endif

        m_keyboard_thread = thread(&keyboard_control::run, this);
        m_started = true;
}

void keyboard_control::stop()
{
        if (!m_started) {
                return;
        }
#ifdef HAVE_TERMIOS_H
        char c = 0;
        assert(write(m_should_exit_pipe[1], &c, 1) == 1);
        close(m_should_exit_pipe[1]);
#else
        m_should_exit = true;
#endif
        m_keyboard_thread.join();
        m_started = false;

#ifdef HAVE_TERMIOS_H
        close(m_should_exit_pipe[0]);
#endif
}

void keyboard_control::run()
{
        while(1) {
#ifdef HAVE_TERMIOS_H
                fd_set set;
                FD_ZERO(&set);
                FD_SET(0, &set);
                FD_SET(m_should_exit_pipe[0], &set);
                select(m_should_exit_pipe[0] + 1, &set, NULL, NULL, NULL);
                if (FD_ISSET(0, &set)) {
                        int c = getchar();
#else
                usleep(200000);
                while (kbhit()) {
                        int c = getch();
#endif
                        bool unknown_key_in_first_switch = false;

                        // This switch processes keys that do not modify UltraGrid
                        // behavior. If some of modifying keys is pressed, warning
                        // is displayed.
                        switch (c) {
                        case CTRL_X:
                                m_locked_against_changes = !m_locked_against_changes; // ctrl-x pressed
                                LOG(LOG_LEVEL_NOTICE) << "Keyboard control: " << (m_locked_against_changes ? "" : "un") << "locked against changes\n";
                                break;
                        case '*':
                        case '/':
                        case '9':
                        case '0':
                        case 'm':
                        case 'M':
                        case '+':
                        case '-':
                        case 'e':
                        case 'v':
                        case 'V':
                                if (m_locked_against_changes) {
                                        LOG(LOG_LEVEL_NOTICE) << "Keyboard control: locked against changes, press 'Ctrl-x' to unlock or 'h' for help.\n";
                                        goto after_protected;
                                } // else process it in next switch
                                break;
                        case 'h':
                                usage();
                                break;
                        case 'i':
                                cout << "\n";
                                info();
                                break;
                        case '\n':
                        case '\r':
                                cout << endl;
                                break;
                        default:
                                unknown_key_in_first_switch = true;
                        }

                        // these are settings that are protected by Ctrl-X
                        switch (c) {
                        case '*':
                        case '/':
                        case '9':
                        case '0':
                        case 'm':
                        {
                                char path[] = "audio.receiver";
                                auto m = (struct msg_receiver *) new_message(sizeof(struct msg_receiver));
                                switch (c) {
                                case '0':
                                case '*': m->type = RECEIVER_MSG_INCREASE_VOLUME; break;
                                case '9':
                                case '/': m->type = RECEIVER_MSG_DECREASE_VOLUME; break;
                                case 'm': m->type = RECEIVER_MSG_MUTE; break;
                                }

                                auto resp = send_message(m_root, path, (struct message *) m);
                                free_response(resp);
                                break;
                        }
                        case 'M':
                        {
                                char path[] = "audio.sender";
                                auto m = (struct msg_sender *) new_message(sizeof(struct msg_sender));
                                m->type = SENDER_MSG_MUTE;

                                auto resp = send_message(m_root, path, (struct message *) m);
                                free_response(resp);
                                break;
                        }
                        case '+':
                        case '-':
                        {
                                int audio_delay = get_audio_delay();
                                audio_delay += c == '+' ? 10 : -10;
                                log_msg(LOG_LEVEL_INFO, "New audio delay: %d ms.\n", audio_delay);
                                set_audio_delay(audio_delay);
                                break;
                        }
                        case 'e':
                        {
                                char path[] = "exporter";
                                auto m = (struct message *) new_message(sizeof(struct msg_universal));
                                strcpy(((struct msg_universal *) m)->text, "toggle");
                                auto resp = send_message(m_root, path, (struct message *) m);
                                free_response(resp);
                                break;
                        }
                        case 'v':
                        case 'V':
                        {
                                if (islower(c)) {
                                        log_level = (log_level + 1) % (LOG_LEVEL_MAX + 1);
                                } else {
                                        log_level = (log_level - 1 + (LOG_LEVEL_MAX + 1)) % (LOG_LEVEL_MAX + 1);
                                }
                                cout << "Log level: " << log_level << "\n";
                                break;
                        }
                        default:
                                if (unknown_key_in_first_switch) {
                                        LOG(LOG_LEVEL_WARNING) << "Keyboard control: Unrecognized key " << c << " pressed. Press 'h' to help.\n";
                                }

                        }
                }

after_protected:
#ifdef HAVE_TERMIOS_H
                if (FD_ISSET(m_should_exit_pipe[0], &set)) {
#else
                if (m_should_exit) {
#endif
                        break;
                }
        }
}

void keyboard_control::info()
{
        cout << "Start time: " << asctime(localtime(&m_start_time));
        cout << "Verbosity level: " << log_level << (log_level == LOG_LEVEL_INFO ? " (default)" : "") << "\n";
        cout << "Locked against changes: " << (m_locked_against_changes ? "true" : "false") << "\n";
        cout << "Audio playback delay: " << get_audio_delay() << " ms\n";

        {
                char path[] = "audio.receiver";
                auto m = (struct msg_receiver *) new_message(sizeof(struct msg_receiver));
                m->type = RECEIVER_MSG_GET_VOLUME;
                auto resp = send_message_sync(m_root, path, (struct message *) m, 100, SEND_MESSAGE_FLAG_QUIET | SEND_MESSAGE_FLAG_NO_STORE);
                if (response_get_status(resp) == 200) {
                        double vol = atof(response_get_text(resp));
                        double db = 20.0 * log10(vol);
                        std::streamsize p = cout.precision();
                        ios_base::fmtflags f = cout.flags();
                        cout << "Received audio volume: " << fixed << setprecision(2) << vol * 100.0 << "% (" << (db >= 0.0 ? "+" : "") <<  db << " dB)\n";
                        cout.precision(p);
                        cout.flags(f);
                }
                free_response(resp);
        }

        {
                char path[] = "audio.sender";
                auto m = (struct msg_sender *) new_message(sizeof(struct msg_sender));
                m->type = SENDER_MSG_GET_STATUS;
                auto resp = send_message_sync(m_root, path, (struct message *) m, 100, SEND_MESSAGE_FLAG_QUIET | SEND_MESSAGE_FLAG_NO_STORE);
                if (response_get_status(resp) == 200) {
                        int muted;
                        sscanf(response_get_text(resp), "%d", &muted);
                        cout << "Sended audio status - muted: " << (bool) muted << "\n";
                }
                free_response(resp);
        }

	{
		struct video_desc desc{};
                struct msg_sender *m = (struct msg_sender *) new_message(sizeof(struct msg_sender));
                m->type = SENDER_MSG_QUERY_VIDEO_MODE;
                struct response *r = send_message_sync(m_root, "sender", (struct message *) m, 100,  SEND_MESSAGE_FLAG_QUIET | SEND_MESSAGE_FLAG_NO_STORE);
                if (response_get_status(r) == RESPONSE_OK) {
                        const char *text = response_get_text(r);
                        istringstream iss(text);
                        iss >> desc;
                        cout << "Captured video format: " <<  desc << "\n";
                }
                free_response(r);
	}

	{
		struct video_desc desc{};
                struct msg_universal *m = (struct msg_universal *) new_message(sizeof(struct msg_universal));
                strcpy(m->text, "get_format");
                struct response *r = send_message_sync(m_root, "receiver.decoder", (struct message *) m, 100,  SEND_MESSAGE_FLAG_QUIET | SEND_MESSAGE_FLAG_NO_STORE);
                if (response_get_status(r) == RESPONSE_OK) {
                        const char *text = response_get_text(r);
                        istringstream iss(text);
                        iss >> desc;
                        cout << "Received video format: " <<  desc << "\n";
                }
                free_response(r);
	}

	{
                struct msg_universal *m = (struct msg_universal *) new_message(sizeof(struct msg_universal));
                strcpy(m->text, "get_port");
                struct response *r = send_message_sync(m_root, "control", (struct message *) m, 100,  SEND_MESSAGE_FLAG_QUIET | SEND_MESSAGE_FLAG_NO_STORE);
                if (response_get_status(r) == RESPONSE_OK) {
                        cout << "Control port: " <<  response_get_text(r) << "\n";
                }
                free_response(r);
	}

	{
                struct msg_universal *m = (struct msg_universal *) new_message(sizeof(struct msg_universal));
                strcpy(m->text, "status");
                struct response *r = send_message_sync(m_root, "exporter", (struct message *) m, 100,  SEND_MESSAGE_FLAG_QUIET | SEND_MESSAGE_FLAG_NO_STORE);
                if (response_get_status(r) == RESPONSE_OK) {
                        cout << "Exporting: " <<  response_get_text(r) << "\n";
                }
                free_response(r);
	}

        cout << "\n";
}

void keyboard_control::usage()
{
        cout << "\nAvailable keybindings:\n" <<
                "\t  * 0  - increase volume\n" <<
                "\t  / 9  - decrease volume\n" <<
                "\t   +   - increase audio delay by 10 ms\n" <<
                "\t   -   - decrease audio delay by 10 ms\n" <<
                "\t   m   - mute/unmute receiver\n" <<
                "\t   M   - mute/unmute sender\n" <<
                "\t   v   - increase verbosity level\n" <<
                "\t   V   - decrease verbosity level\n" <<
                "\t   e   - record captured content (toggle)\n" <<
                "\t   h   - show help\n" <<
                "\t   i   - show various information\n" <<
                "\tCtrl-x - unlock/lock against changes\n" <<
                "\tCtrl-c - exit\n" <<
                "\n";
}

