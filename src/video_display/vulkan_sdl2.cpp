/**
 * @file   video_display/sdl2.cpp
 * @author Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 * @author Milos Liska      <xliska@fi.muni.cz>
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
 /*
  * Copyright (c) 2018-2021 CESNET, z. s. p. o.
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
  /**
   * @file
   * @todo
   * Missing from SDL1:
   * * audio (would be perhaps better as an audio playback device)
   * @todo
   * * frames are copied, better would be to preallocate textures and set
   *   video_frame::tiles::data to SDL_LockTexture() pixels. This, however,
   *   needs decoder to use either pitch (toggling fullscreen or resize) or
   *   forcing decoder to reconfigure pitch.
   */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H


   //#define NO_EXCEPTIONS
#define VK_USE_PLATFORM_WIN32_KHR
#include "vulkan_display.h" // Vulkan.h must be before GLFW/SDL

#include "debug.h"
#include "host.h"
#include "keyboard_control.h" // K_*
#include "lib_common.h"
#include "messaging.h"
#include "module.h"
#include "rang.hpp"
#include "video_display.h"
#include "video_display/splashscreen.h"
#include "video.h"

/// @todo remove the defines when no longer needed
#ifdef __arm64__
#define SDL_DISABLE_MMINTRIN_H 1
#define SDL_DISABLE_IMMINTRIN_H 1
#endif // defined __arm64__
#include <SDL2/SDL.h>

#include <SDL2/SDL_vulkan.h>
#include <SDL2/SDL_syswm.h>

#include <array>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <queue>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility> // pair

using rang::fg;
using rang::style;

namespace chrono = std::chrono;
using namespace std::string_literals;

namespace {

constexpr int MAGIC_VULKAN_SDL2 = 0x3cc234a2;
constexpr int MAX_BUFFER_SIZE = 1;
#define MOD_NAME "[VULKAN_SDL2] "


void display_sdl2_new_message(module*);
int display_sdl2_putf(void* state, video_frame* frame, int nonblock);

class window_callback final : public window_changed_callback {
        SDL_Window* window = nullptr;
public:
        window_callback(SDL_Window* window):
                window{window} { }
        
        window_parameters get_window_parameters() override {
                assert(window);
                int width, height;
                SDL_Vulkan_GetDrawableSize(window, &width, &height);
                if (SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED) {
                        width = 0;
                        height = 0;
                }
                return { static_cast<uint32_t>(width), static_cast<uint32_t>(height), true };
        }
};


struct state_vulkan_sdl2 {
        module                  mod;

        Uint32                  sdl_user_new_frame_event;
        Uint32                  sdl_user_new_message_event;

        chrono::steady_clock::time_point tv{ chrono::steady_clock::now() };
        unsigned long long      frames{ 0 };

        int                     display_idx{ 0 };
        int                     x{ SDL_WINDOWPOS_UNDEFINED };
        int                     y{ SDL_WINDOWPOS_UNDEFINED };

        SDL_Window*             window{ nullptr };


        uint32_t                gpu_idx{ NO_GPU_SELECTED };
        bool                    validation{ true }; // todo: change to false

        bool                    fs{ false };
        bool                    deinterlace{ false };
        bool                    keep_aspect{ false };
        bool                    vsync{ true };
        bool                    fixed_size{ false };
        int                     fixed_w{ 0 }, fixed_h{ 0 };
        uint32_t                window_flags{ 0 }; ///< user requested flags


        std::mutex              lock;
        std::condition_variable frame_consumed_cv;
        int                     buffered_frames_count{ 0 };

        video_desc              current_desc{};
        video_desc              current_display_desc{};
        video_frame*            last_frame{ nullptr };

        std::queue<video_frame*> free_frame_queue;

        std::unique_ptr<vulkan_display> vulkan = nullptr;

        window_callback window_callback { nullptr };

        state_vulkan_sdl2(module* parent) {
                module_init_default(&mod);
                mod.priv_magic = MAGIC_VULKAN_SDL2;
                mod.new_message = display_sdl2_new_message;
                mod.cls = MODULE_CLASS_DATA;
                module_register(&mod, parent);

                sdl_user_new_frame_event = SDL_RegisterEvents(2);
                assert(sdl_user_new_frame_event != (Uint32)-1);
                sdl_user_new_message_event = sdl_user_new_frame_event + 1;
        }

        ~state_vulkan_sdl2() {
                module_done(&mod);
        }
};


constexpr std::array<std::pair<char, std::string_view>, 3> display_sdl2_keybindings{{
        {'d', "toggle deinterlace"},
        {'f', "toggle fullscreen"},
        {'q', "quit"}
}};


void display_frame(state_vulkan_sdl2* s, video_frame* frame) {
        if (!frame) {
                return;
        }
        /*if (!video_desc_eq(video_desc_from_frame(frame), s->current_display_desc)) {
            if (!display_sdl2_reconfigure_real(s, video_desc_from_frame(frame))) {
                        goto free_frame;
            }
        }

        if (!s->deinterlace) {
                int pitch;
                if (codec_is_planar(frame->color_spec)) {
                        pitch = frame->tiles[0].width;
                } else {
                        pitch = vc_get_linesize(frame->tiles[0].width, frame->color_spec);
                }
                //SDL_UpdateTexture(s->texture, NULL, frame->tiles[0].data, pitch);
        }
        else {
                //unsigned char *pixels;
                //int pitch;
                //SDL_LockTexture(s->texture, NULL, (void **) &pixels, &pitch);
                //vc_deinterlace_ex();

                //SDL_UnlockTexture(s->texture);

        }*/
        if (s->deinterlace) {
                vc_deinterlace(reinterpret_cast<unsigned char*>(frame->tiles[0].data),
                        vc_get_linesize(frame->tiles[0].width, frame->color_spec), frame->tiles[0].height);
        }
        try {
                s->vulkan->render(reinterpret_cast<std::byte*>(frame->tiles[0].data),
                        frame->tiles[0].width, frame->tiles[0].height,
                        frame->color_spec == RGBA ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8Srgb);
        }
        catch (std::exception& e) {
                LOG(LOG_LEVEL_ERROR) << e.what() << std::endl;
                exit_uv(EXIT_FAILURE);
        }
        //SDL_RenderCopy(s->renderer, s->texture, NULL, NULL);
        //SDL_RenderPresent(s->renderer);

//free_frame:

        if (frame == s->last_frame) {
                return; // we are only redrawing on window resize
        }

        if (s->last_frame) {
                s->lock.lock();
                s->free_frame_queue.push(s->last_frame);
                s->lock.unlock();
        }
        s->last_frame = frame;

        s->frames++;
        auto tv = chrono::steady_clock::now();
        double seconds = chrono::duration_cast<chrono::duration<double>>(tv - s->tv).count();
        if (seconds > 5) {
                double fps = s->frames / seconds;
                log_msg(LOG_LEVEL_INFO, "[SDL] %llu frames in %g seconds = %g FPS\n",
                        s->frames, seconds, fps);
                s->tv = tv;
                s->frames = 0;
        }
}


int64_t translate_sdl_key_to_ug(SDL_Keysym sym) {
        sym.mod &= ~(KMOD_NUM | KMOD_CAPS); // remove num+caps lock modifiers

        // ctrl alone -> do not interpret
        if (sym.sym == SDLK_LCTRL || sym.sym == SDLK_RCTRL) {
                return 0;
        }

        bool ctrl = false;
        bool shift = false;
        if (sym.mod & KMOD_CTRL) {
                ctrl = true;
        }
        sym.mod &= ~KMOD_CTRL;

        if (sym.mod & KMOD_SHIFT) {
                shift = true;
        }
        sym.mod &= ~KMOD_SHIFT;

        if (sym.mod != 0) {
                return -1;
        }

        if ((sym.sym & SDLK_SCANCODE_MASK) == 0) {
                if (shift) {
                        sym.sym = toupper(sym.sym);
                }
                return ctrl ? K_CTRL(sym.sym) : sym.sym;
        }
        switch (sym.sym) {
        case SDLK_RIGHT: return K_RIGHT;
        case SDLK_LEFT:  return K_LEFT;
        case SDLK_DOWN:  return K_DOWN;
        case SDLK_UP:    return K_UP;
        case SDLK_PAGEDOWN:    return K_PGDOWN;
        case SDLK_PAGEUP:    return K_PGUP;
        }
        return -1;
}

bool display_sdl2_process_key(state_vulkan_sdl2* s, int64_t key) {
        switch (key) {
        case 'd':
                s->deinterlace = !s->deinterlace;
                log_msg(LOG_LEVEL_INFO, "Deinterlacing: %s\n",
                        s->deinterlace ? "ON" : "OFF");
                return true;
        case 'f':
                s->fs = !s->fs;
                SDL_SetWindowFullscreen(s->window, s->fs ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0);
                return true;
        case 'q':
                exit_uv(0);
                return true;
        default:
                return false;
        }
}

void display_sdl2_run(void* state) {
        auto s = static_cast<state_vulkan_sdl2*>(state);
        assert(s->mod.priv_magic == MAGIC_VULKAN_SDL2);
        bool should_exit_sdl = false;

        while (!should_exit_sdl) {
                SDL_Event sdl_event;
                if (!SDL_WaitEvent(&sdl_event)) {
                        continue;
                }
                if (sdl_event.type == s->sdl_user_new_frame_event) {
                        std::unique_lock<std::mutex> lk(s->lock);
                        //LOG(LOG_LEVEL_INFO) << "Buffered frames count:" << s->buffered_frames_count << '\n';
                        s->buffered_frames_count -= 1;
                        lk.unlock();
                        s->frame_consumed_cv.notify_one();
                        if (sdl_event.user.data1 != NULL) {
                                display_frame(s, static_cast<video_frame*>(sdl_event.user.data1));
                        } else { // poison pill received
                                should_exit_sdl = true;
                        }
                } else if (sdl_event.type == s->sdl_user_new_message_event) {
                        msg_universal* msg;
                        while ((msg = reinterpret_cast<msg_universal*>(check_message(&s->mod)))) {
                                log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Received message: %s\n", msg->text);
                                response* r;
                                int key;
                                if (strstr(msg->text, "win-title ") == msg->text) {
                                        SDL_SetWindowTitle(s->window, msg->text + strlen("win-title "));
                                        r = new_response(RESPONSE_OK, NULL);
                                } else if (sscanf(msg->text, "%d", &key) == 1) {
                                        if (!display_sdl2_process_key(s, key)) {
                                                r = new_response(RESPONSE_BAD_REQUEST, "Unsupported key for SDL");
                                        } else {
                                                r = new_response(RESPONSE_OK, NULL);
                                        }
                                } else {
                                        r = new_response(RESPONSE_BAD_REQUEST, "Wrong command");
                                }

                                free_message(reinterpret_cast<message*>(msg), r);
                        }
                } else if (sdl_event.type == SDL_KEYDOWN) {
                        log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Pressed key %s (scancode: %d, sym: %d, mod: %d)!\n",
                                SDL_GetKeyName(sdl_event.key.keysym.sym), sdl_event.key.keysym.scancode, sdl_event.key.keysym.sym, sdl_event.key.keysym.mod);
                        int64_t sym = translate_sdl_key_to_ug(sdl_event.key.keysym);
                        if (sym > 0) {
                                if (!display_sdl2_process_key(s, sym)) { // unknown key -> pass to control
                                        keycontrol_send_key(get_root_module(&s->mod), sym);
                                }
                        } else if (sym == -1) {
                                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Cannot translate key %s (scancode: %d, sym: %d, mod: %d)!\n",
                                        SDL_GetKeyName(sdl_event.key.keysym.sym), sdl_event.key.keysym.scancode, sdl_event.key.keysym.sym, sdl_event.key.keysym.mod);
                        }
                } else if (sdl_event.type == SDL_WINDOWEVENT) {
                        // https://forums.libsdl.org/viewtopic.php?p=38342
                        if (s->keep_aspect && sdl_event.window.event == SDL_WINDOWEVENT_RESIZED) {
                                double area = sdl_event.window.data1 * sdl_event.window.data2;
                                int width = sqrt(area / ((double)s->current_display_desc.height / s->current_display_desc.width));
                                int height = sqrt(area / ((double)s->current_display_desc.width / s->current_display_desc.height));
                                SDL_SetWindowSize(s->window, width, height);
                                debug_msg("[SDL] resizing to %d x %d\n", width, height);
                        } else if (sdl_event.window.event == SDL_WINDOWEVENT_EXPOSED
                                || sdl_event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED)
                        {
                                display_frame(s, s->last_frame);
                        }
                } else if (sdl_event.type == SDL_QUIT) {
                        exit_uv(0);
                }
        }
}

void sdl2_print_displays() {
        for (int i = 0; i < SDL_GetNumVideoDisplays(); ++i) {
                if (i > 0) {
                        std::cout << ", ";
                }
                const char* dname = SDL_GetDisplayName(i);
                if (dname == nullptr) {
                        dname = SDL_GetError();
                }
                std::cout << style::bold << i << style::reset << " - " << dname;
        }
        std::cout << "\n";
}

void print_gpus() {
        vulkan_display vulkan;
        std::vector<const char*>required_extensions{};
        vulkan.create_instance(required_extensions, false);

        std::cout << "\n\tVulkan GPUs:\n";

        std::vector<std::pair<std::string, bool>> gpus;
        vulkan.get_available_gpus(gpus);
        gpus[1].second = false;
        uint32_t counter = 0;
        for (const auto& gpu : gpus) {
                std::cout << (gpu.second ? fg::reset : fg::red);
                std::cout << "\t\t " << counter++ << "\t - " << gpu.first;
                std::cout << (gpu.second ? "" : " - unsuitable") << fg::reset << '\n';
        }
}

void show_help() {
        SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS);
        auto& cout = std::cout;
        cout << "SDL options:\n";
        cout << style::bold << fg::red << "\t-d vulkan_sdl2" << fg::reset;
        cout << "[:fs|:d|:display=<dis_id>|:novsync|:nodecorate|:fixed_size[=WxH]|:window_flags=<f>|:pos=<x>,<y>|:keep-aspect|:validation|:gpu=<gpu_id>|:help]\n";

        cout << style::reset << ("\twhere:\n");

        cout << style::bold << "\t\t       d" << style::reset << " - deinterlace\n";
        cout << style::bold << "\t\t      fs" << style::reset << " - fullscreen\n";
        cout << style::bold << "\t\t<dis_id>" << style::reset << " - display index, available indices: ";
        sdl2_print_displays();
        cout << style::bold << "\t     keep-aspect" << style::reset << " - keep window aspect ratio respecive to the video\n";
        cout << style::bold << "\t         novsync" << style::reset << " - disable vsync\n";
        cout << style::bold << "\t      nodecorate" << style::reset << " - disable window border\n";
        cout << style::bold << "\tfixed_size[=WxH]" << style::reset << " - use fixed sized window\n";
        cout << style::bold << "\t    window_flags" << style::reset << " - flags to be passed to SDL_CreateWindow (use prefix 0x for hex)\n";
        cout << style::bold << "\t      validation" << style::reset << " - enable vulkan validation layers\n";
        cout << style::bold << "\t        <gpu_id>" << style::reset << " - gpu index selected from the following list\n";
        print_gpus();

        cout << "\n\tKeyboard shortcuts:\n";
        for (auto& binding : display_sdl2_keybindings) {
                cout << style::bold << "\t\t'" << binding.first << 
                        style::reset << "'\t - " << binding.second << "\n";
        }
        SDL_Quit();
}

int display_sdl2_reconfigure(void* state, video_desc desc) {
        auto s = static_cast<state_vulkan_sdl2*>(state);
        assert(s->mod.priv_magic == MAGIC_VULKAN_SDL2);

        s->current_desc = desc;
        return 1;
}

struct ug_to_sdl_pf {
        codec_t first;
        uint32_t second;
};

constexpr std::array<ug_to_sdl_pf, 2> pf_mapping{{
        //{ I420, SDL_PIXELFORMAT_IYUV },
        //{ UYVY, SDL_PIXELFORMAT_UYVY },
        //{ YUYV, SDL_PIXELFORMAT_YUY2 },
        //{ BGR, SDL_PIXELFORMAT_BGR24 },
        { RGB, SDL_PIXELFORMAT_RGB24 },
        { RGBA, SDL_PIXELFORMAT_RGBA32 },
}};

/*uint32_t get_ug_to_sdl_format(codec_t ug_codec) {
        if (ug_codec == R10k) {
                return SDL_PIXELFORMAT_ARGB2101010;
        }

        const auto *it = find_if(begin(pf_mapping), end(pf_mapping), [ug_codec](const ug_to_sdl_pf &u) { return u.first == ug_codec; });
        if (it == end(pf_mapping)) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Wrong codec: " << get_codec_name(ug_codec) << "\n";
                return SDL_PIXELFORMAT_UNKNOWN;
        }
        return it->second;
}*/

auto get_supported_pfs() {
        std::vector<codec_t> codecs;
        codecs.reserve(sizeof pf_mapping / sizeof pf_mapping[0]);

        for (auto item : pf_mapping) {
                codecs.push_back(item.first);
        }

        return codecs;
}

/**
 * Load splashscreen
 * Function loads graphic data from header file "splashscreen.h", where are
 * stored splashscreen data in RGB format.
 */
void loadSplashscreen(state_vulkan_sdl2* s) {
        video_desc desc{};

        desc.width = 512;
        desc.height = 512;
        desc.color_spec = RGBA;
        desc.interlacing = PROGRESSIVE;
        desc.fps = 1;
        desc.tile_count = 1;

        display_sdl2_reconfigure(s, desc);

        video_frame* frame = vf_alloc_desc_data(desc);

        const char* data = splash_data;
        memset(frame->tiles[0].data, 0, frame->tiles[0].data_len);
        for (unsigned int y = 0; y < splash_height; ++y) {
                char* line = frame->tiles[0].data;
                line += vc_get_linesize(frame->tiles[0].width,
                        frame->color_spec) *
                        (((frame->tiles[0].height - splash_height) / 2) + y);
                line += vc_get_linesize(
                        (frame->tiles[0].width - splash_width) / 2,
                        frame->color_spec);
                for (unsigned int x = 0; x < splash_width; ++x) {
                        HEADER_PIXEL(data, line);
                        line += 4;
                }
        }

        display_sdl2_putf(s, frame, PUTF_BLOCKING);
}

void* display_sdl2_init(module* parent, const char* fmt, unsigned int flags) {
        if (flags & DISPLAY_FLAG_AUDIO_ANY) {
                log_msg(LOG_LEVEL_ERROR, "UltraGrid VULKAN_SDL2 module currently doesn't support audio!\n");
                return NULL;
        }

        auto s = new state_vulkan_sdl2{ parent };

        if (fmt == NULL) {
                fmt = "";
        }
        char* tmp = (char*)alloca(strlen(fmt) + 1);
        strcpy(tmp, fmt);
        char* tok, * save_ptr;
        while ((tok = strtok_r(tmp, ":", &save_ptr)))
        {
                if (strcmp(tok, "d") == 0) {
                        s->deinterlace = true;
                } else if (strncmp(tok, "display=", strlen("display=")) == 0) {
                        s->display_idx = atoi(tok + strlen("display="));
                } else if (strcmp(tok, "fs") == 0) {
                        s->fs = true;
                } else if (strcmp(tok, "help") == 0) {
                        show_help();
                        delete s;
                        return &display_init_noerr;
                } else if (strcmp(tok, "novsync") == 0) {
                        s->vsync = false;
                } else if (strcmp(tok, "nodecorate") == 0) {
                        s->window_flags |= SDL_WINDOW_BORDERLESS;
                } else if (strcmp(tok, "keep-aspect") == 0) {
                        s->keep_aspect = true;
                } else if (strcmp(tok, "validation") == 0) {
                        s->validation = true;
                } else if (strncmp(tok, "fixed_size", strlen("fixed_size")) == 0) {
                        s->fixed_size = true;
                        if (strncmp(tok, "fixed_size=", strlen("fixed_size=")) == 0) {
                                char* size = tok + strlen("fixed_size=");
                                if (strchr(size, 'x')) {
                                        s->fixed_w = atoi(size);
                                        s->fixed_h = atoi(strchr(size, 'x') + 1);
                                }
                        }
                } else if (strstr(tok, "window_flags=") == tok) {
                        int f;
                        if (sscanf(tok + strlen("window_flags="), "%i", &f) != 1) {
                                log_msg(LOG_LEVEL_ERROR, "Wrong window_flags: %s\n", tok);
                                delete s;
                                return NULL;
                        }
                        s->window_flags |= f;
                } else if (strstr(tok, "pos=") == tok) {
                        tok += strlen("pos=");
                        if (strchr(tok, ',') == nullptr) {
                                log_msg(LOG_LEVEL_ERROR, "[SDL] position: %s\n", tok);
                                delete s;
                                return NULL;
                        }
                        s->x = atoi(tok);
                        s->y = atoi(strchr(tok, ',') + 1);
                } else if (strncmp(tok, "gpu=", strlen("gpu=")) == 0) {
                        s->gpu_idx = atoi(tok + strlen("gpu="));
                } else {
                        log_msg(LOG_LEVEL_ERROR, "[SDL] Wrong option: %s\n", tok);
                        delete s;
                        return NULL;
                }
                tmp = NULL;
        }


        int ret = SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS);
        if (ret < 0) {
                log_msg(LOG_LEVEL_ERROR, "Unable to initialize SDL2: %s\n", SDL_GetError());
                delete s;
                return NULL;
        }

        //SDL_ShowCursor(SDL_DISABLE);
        SDL_DisableScreenSaver();

        loadSplashscreen(s);
        for (auto& binding : display_sdl2_keybindings) {
                std::string msg = std::to_string(static_cast<int>(binding.first));
                keycontrol_register_key(&s->mod, binding.first, msg.c_str(), binding.second.data());
        }

        log_msg(LOG_LEVEL_NOTICE, "SDL2 initialized successfully.\n");



        const char* window_title = "UltraGrid - SDL2 Display";
        if (get_commandline_param("window-title")) {
                window_title = get_commandline_param("window-title");
        }

        int x = s->x == SDL_WINDOWPOS_UNDEFINED ? SDL_WINDOWPOS_CENTERED_DISPLAY(s->display_idx) : s->x;
        int y = s->y == SDL_WINDOWPOS_UNDEFINED ? SDL_WINDOWPOS_CENTERED_DISPLAY(s->display_idx) : s->y;
        int width = s->fixed_w ? s->fixed_w : 960;
        int height = s->fixed_h ? s->fixed_h : 540;

        int window_flags = s->window_flags | SDL_WINDOW_RESIZABLE | SDL_WINDOW_VULKAN;
        if (s->fs) {
                window_flags |= SDL_WINDOW_FULLSCREEN_DESKTOP;
        }

        s->window = SDL_CreateWindow(window_title, x, y, width, height, window_flags);
        if (!s->window) {
                log_msg(LOG_LEVEL_ERROR, "[SDL] Unable to create window: %s\n", SDL_GetError());
                return nullptr;
        }
        s->window_callback = ::window_callback{ s->window };

        uint32_t extension_count = 0;
        SDL_Vulkan_GetInstanceExtensions(s->window, &extension_count, nullptr);
        std::vector<const char*> required_extensions(extension_count);
        SDL_Vulkan_GetInstanceExtensions(s->window, &extension_count, required_extensions.data());
        assert(extension_count > 0);
        try {
                s->vulkan = std::make_unique<vulkan_display>();
                s->vulkan->create_instance(required_extensions, s->validation);
                const auto& instance = s->vulkan->get_instance();

#ifdef __MINGW32__
                //SDL2 for MINGW has problem creating surface
                SDL_SysWMinfo wmInfo;
                SDL_VERSION(&wmInfo.version);
                SDL_GetWindowWMInfo(s->window, &wmInfo);
                HWND hwnd = wmInfo.info.win.window;
                HINSTANCE hinst = wmInfo.info.win.hinstance;
                vk::SurfaceKHR surface;
                auto const createInfo = vk::Win32SurfaceCreateInfoKHR{}.setHinstance(hinst).setHwnd(hwnd);
                if (instance.createWin32SurfaceKHR(&createInfo, nullptr, &surface) != vk::Result::eSuccess) {
                        throw std::runtime_error("Surface cannot be created.");
                }
#else
                VkSurfaceKHR surface = nullptr;
                if (!SDL_Vulkan_CreateSurface(s->window, instance, &surface)) {
                        std::cout << SDL_GetError() << std::endl;
                        assert(false);
                        throw std::runtime_error("SDL cannot create surface.");
                }
#endif

                s->vulkan->init(surface, &s->window_callback, s->gpu_idx);
                LOG(LOG_LEVEL_NOTICE) << "Vulkan display initialised." << std::endl;
        }
        catch (std::exception& e) {
                LOG(LOG_LEVEL_ERROR) << e.what() << "\n";
                return nullptr;
        }
        return (void*)s;
}

void display_sdl2_done(void* state) {
        auto s = static_cast<state_vulkan_sdl2*>(state);
        assert(s->mod.priv_magic == MAGIC_VULKAN_SDL2);

        SDL_ShowCursor(SDL_ENABLE);

        vf_free(s->last_frame);

        while (s->free_frame_queue.size() > 0) {
                video_frame* buffer = s->free_frame_queue.front();
                s->free_frame_queue.pop();
                vf_free(buffer);
        }

        s->vulkan = nullptr;

        if (s->window) {
                SDL_DestroyWindow(s->window);
                s->window = nullptr;
        }


        SDL_QuitSubSystem(SDL_INIT_EVENTS);
        SDL_Quit();

        delete s;
}

video_frame* display_sdl2_getf(void* state) {
        auto s = static_cast<state_vulkan_sdl2*>(state);
        assert(s->mod.priv_magic == MAGIC_VULKAN_SDL2);

        std::scoped_lock lock(s->lock);

        while (s->free_frame_queue.size() > 0) {
                video_frame* buffer = s->free_frame_queue.front();
                s->free_frame_queue.pop();
                if (video_desc_eq(video_desc_from_frame(buffer), s->current_desc)) {
                        return buffer;
                } else {
                        vf_free(buffer);
                }
        }

        return vf_alloc_desc_data(s->current_desc);
}

int display_sdl2_putf(void* state, video_frame* frame, int nonblock) {
        auto s = static_cast<state_vulkan_sdl2*>(state);
        assert(s->mod.priv_magic == MAGIC_VULKAN_SDL2);

        std::unique_lock<std::mutex> lk(s->lock);
        if (nonblock == PUTF_DISCARD) {
                assert(frame != nullptr);
                s->free_frame_queue.push(frame);
                return 0;
        }

        if (s->buffered_frames_count >= MAX_BUFFER_SIZE && nonblock == PUTF_NONBLOCK
                && frame != NULL) {
                s->free_frame_queue.push(frame);
                LOG(LOG_LEVEL_INFO) << MOD_NAME << "1 frame(s) dropped!\n";
                return 1;
        }
        auto not_full_queue = [s] {return s->buffered_frames_count < MAX_BUFFER_SIZE; };
        auto success = s->frame_consumed_cv.wait_for(lk, chrono::seconds(3), not_full_queue);
        if (!success) {
                log_msg(LOG_LEVEL_ERROR, "Frame cannot be put in the queue, because the queue is full.\n");
        }
        s->buffered_frames_count += 1;
        lk.unlock();
        SDL_Event event{};
        event.type = s->sdl_user_new_frame_event;
        event.user.data1 = frame;
        SDL_PushEvent(&event);

        return 0;
}

int display_sdl2_get_property([[maybe_unused]] void* state, int property, void* val, size_t* len) {
        auto codecs = get_supported_pfs();
        size_t codecs_len = codecs.size() * sizeof(codec_t);

        switch (property) {
        case DISPLAY_PROPERTY_CODECS:
                if (codecs_len <= *len) {
                        memcpy(val, codecs.data(), codecs_len);
                        *len = codecs_len;
                } else {
                        return FALSE;
                }
                break;
        default:
                return FALSE;
        }
        return TRUE;
}

void display_sdl2_new_message(module* mod) {
        auto s = reinterpret_cast<state_vulkan_sdl2*>(mod);
        assert(s->mod.priv_magic == MAGIC_VULKAN_SDL2);

        SDL_Event event{};
        event.type = s->sdl_user_new_message_event;
        SDL_PushEvent(&event);
}

void display_sdl2_put_audio_frame([[maybe_unused]] void* state, [[maybe_unused]] audio_frame* frame){}

int display_sdl2_reconfigure_audio([[maybe_unused]] void* state, [[maybe_unused]] int quant_samples,
        [[maybe_unused]] int channels, [[maybe_unused]] int sample_rate)
{
        return FALSE;
}

const video_display_info display_sdl2_info = {
        [](device_info** available_cards, int* count, void (**deleter)(void*)) {
                UNUSED(deleter);
                *count = 1;
                *available_cards = (device_info*)calloc(1, sizeof(device_info));
                strcpy((*available_cards)[0].dev, "");
                strcpy((*available_cards)[0].name, "VULKAN_SDL2 SW display");
                (*available_cards)[0].repeatable = true;
        },
        display_sdl2_init,
        display_sdl2_run,
        display_sdl2_done,
        display_sdl2_getf,
        display_sdl2_putf,
        display_sdl2_reconfigure,
        display_sdl2_get_property,
        display_sdl2_put_audio_frame,
        display_sdl2_reconfigure_audio,
        DISPLAY_NEEDS_MAINLOOP,
};

} // namespace


REGISTER_MODULE(vulkan_sdl2, &display_sdl2_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);

