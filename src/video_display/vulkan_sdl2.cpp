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
#include <atomic>
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


namespace vkd = vulkan_display;
namespace chrono = std::chrono;
using namespace std::string_literals;

namespace {

constexpr int MAGIC_VULKAN_SDL2 = 0x3cc234a2;
constexpr int MAX_FRAME_COUNT = 5;
#define MOD_NAME "[VULKAN_SDL2] "


void display_sdl2_new_message(module*);
int display_sdl2_putf(void* state, video_frame* frame, int nonblock);
video_frame* display_sdl2_getf(void* state);

class window_callback final : public vkd::window_changed_callback {
        SDL_Window* window = nullptr;
public:
        window_callback(SDL_Window* window):
                window{window} { }
        
        vkd::window_parameters get_window_parameters() override {
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

        Uint32                  sdl_user_new_message_event;

        chrono::steady_clock::time_point tv{ chrono::steady_clock::now() };
        unsigned long long      frames{ 0 };

        int                     display_idx{ 0 };
        int                     x{ SDL_WINDOWPOS_UNDEFINED };
        int                     y{ SDL_WINDOWPOS_UNDEFINED };

        SDL_Window*             window{ nullptr };


        uint32_t                gpu_idx{ vkd::NO_GPU_SELECTED };
        bool                    validation{ false };//todo change to false

        bool                    fs{ false };
        bool                    deinterlace{ false };
        bool                    keep_aspect{ false };
        bool                    vsync{ true };
        bool                    fixed_size{ false };
        int                     fixed_w{ 0 }, fixed_h{ 0 };
        uint32_t                window_flags{ 0 }; ///< user requested flags

        std::atomic<bool>       should_exit = false;

        video_desc              current_desc{};
        video_desc              current_display_desc{};


        std::array<video_frame, MAX_FRAME_COUNT> video_frames {};
        std::array<vkd::image, MAX_FRAME_COUNT> images {};

        std::unique_ptr<vkd::vulkan_display> vulkan = nullptr;

        window_callback window_callback { nullptr };

        state_vulkan_sdl2(module* parent) {
                module_init_default(&mod);
                mod.priv_magic = MAGIC_VULKAN_SDL2;
                mod.new_message = display_sdl2_new_message;
                mod.cls = MODULE_CLASS_DATA;
                module_register(&mod, parent);

                sdl_user_new_message_event = SDL_RegisterEvents(1);
                assert(sdl_user_new_message_event != static_cast<Uint32>(-1));
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

void update_description(const video_desc& video_desc, video_frame& frame) {
        frame.color_spec = video_desc.color_spec;
        frame.fps = video_desc.fps;
        frame.interlacing = video_desc.interlacing;
        frame.tile_count = video_desc.tile_count;
        for (unsigned i = 0; i < video_desc.tile_count; i++) {
                frame.tiles[i].width = video_desc.width;
                frame.tiles[i].height = video_desc.height;
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

void log_and_exit(std::exception e) {
        LOG(LOG_LEVEL_ERROR) << e.what() << std::endl;
        exit_uv(EXIT_FAILURE);
}

void display_sdl2_run(void* state) {
        auto s = static_cast<state_vulkan_sdl2*>(state);
        assert(s->mod.priv_magic == MAGIC_VULKAN_SDL2);


        while (!s->should_exit) {
                SDL_Event sdl_event;
                while (SDL_PollEvent(&sdl_event)) {
                        if (sdl_event.type == s->sdl_user_new_message_event) {
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
                                        debug_msg(MOD_NAME "resizing to %d x %d\n", width, height);
                                } else if (sdl_event.window.event == SDL_WINDOWEVENT_EXPOSED
                                        || sdl_event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED)
                                {
                                        //todo support last frame redrawing
                                }
                        } else if (sdl_event.type == SDL_QUIT) {
                                exit_uv(0);
                        }
                }
                try {
                        s->vulkan->display_queued_image();
                } catch (std::exception& e) { log_and_exit(e); }
                s->frames++;
                auto tv = chrono::steady_clock::now();
                double seconds = chrono::duration_cast<chrono::duration<double>>(tv - s->tv).count();
                if (seconds > 5) {
                        double fps = s->frames / seconds;
                        log_msg(LOG_LEVEL_INFO, MOD_NAME "%llu frames in %g seconds = %g FPS\n",
                                s->frames, seconds, fps);
                        s->tv = tv;
                        s->frames = 0;
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
        vkd::vulkan_display vulkan;
        std::vector<const char*>required_extensions{};
        std::vector<std::pair<std::string, bool>> gpus;
        try {
                vulkan.create_instance(required_extensions, false);
                vulkan.get_available_gpus(gpus);
        } catch (std::exception& e) { log_and_exit(e); }
        
        std::cout << "\n\tVulkan GPUs:\n";
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
        cout << "VULKAN_SDL2 options:\n";
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

        assert(desc.tile_count == 1);
        s->current_desc = desc;
        return 1;
}

/**
 * Load splashscreen
 * Function loads graphic data from header file "splashscreen.h", where are
 * stored splashscreen data in RGB format.
 */
void draw_splashscreen(state_vulkan_sdl2* s) {
        vkd::image image;
        s->vulkan->acquire_image(image, {splash_width, splash_height, vk::Format::eR8G8B8A8Srgb});

        const char* source = splash_data;
        char* dest = reinterpret_cast<char*>(image.get_memory_ptr());
        auto padding = image.get_row_pitch() - splash_width * 4;

        for (unsigned row = 0; row < splash_height; row++) {
                for (unsigned int col = 0; col < splash_width; col++) {
                        HEADER_PIXEL(source, dest);
                        dest += 4;
                }
                dest += padding;
        }

        s->vulkan->queue_image(image);
        s->vulkan->display_queued_image();
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
        char* tmp = static_cast<char*>(alloca(strlen(fmt) + 1));
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
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to create window : %s\n", SDL_GetError());
                return nullptr;
        }
        s->window_callback = ::window_callback{ s->window };

        uint32_t extension_count = 0;
        SDL_Vulkan_GetInstanceExtensions(s->window, &extension_count, nullptr);
        std::vector<const char*> required_extensions(extension_count);
        SDL_Vulkan_GetInstanceExtensions(s->window, &extension_count, required_extensions.data());
        assert(extension_count > 0);
        try {
                s->vulkan = std::make_unique<vkd::vulkan_display>();
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

                s->vulkan->init(surface, MAX_FRAME_COUNT, &s->window_callback, s->gpu_idx);
                LOG(LOG_LEVEL_NOTICE) << MOD_NAME "Vulkan display initialised." << std::endl;
        }
        catch (std::exception& e) { log_and_exit(e); }

        for (auto& frame : s->video_frames) {
                frame = video_frame{};
        }

        draw_splashscreen(s);
        return (void*)s;
}

void display_sdl2_done(void* state) {
        auto s = static_cast<state_vulkan_sdl2*>(state);
        assert(s->mod.priv_magic == MAGIC_VULKAN_SDL2);

        SDL_ShowCursor(SDL_ENABLE);

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
        
        const auto& desc = s->current_desc;
        vulkan_display::image image;
        try {
                s->vulkan->acquire_image(image, { desc.width, desc.height });
        } catch (std::exception& e) { log_and_exit(e); }
        s->images[image.get_id()] = image;
        
        video_frame& frame = s->video_frames[image.get_id()];
        update_description(desc, frame);
        frame.tiles[0].data_len = image.get_row_pitch() * image.get_size().height;
        frame.tiles[0].data = reinterpret_cast<char*>(image.get_memory_ptr());
        return &frame;
}

int display_sdl2_putf(void* state, video_frame* frame, int nonblock) {
        auto s = static_cast<state_vulkan_sdl2*>(state);
        assert(s->mod.priv_magic == MAGIC_VULKAN_SDL2);

        uint32_t id = std::distance(s->video_frames.data(), frame);
        auto& image = s->images[id];
        if (nonblock == PUTF_DISCARD) {
                assert(frame != nullptr);
                assert(image.get_id() == id);
                try {
                        s->vulkan->discard_image(s->images[id]);
                } catch (std::exception& e) { log_and_exit(e); }
                return 0;
        }

        if (!frame) {
                s->should_exit = true;
                s->vulkan->queue_image(vkd::image{});
                return 0;
        }
        assert(image.get_id() == id);
        s->current_desc = video_desc_from_frame(frame);
        

        if (s->deinterlace) {
                image.set_process_function([](vkd::image& image) {
                        vc_deinterlace(reinterpret_cast<unsigned char*>(image.get_memory_ptr()),
                                image.get_row_pitch(),
                                image.get_size().height);
                });
        }
        
        try {
                s->vulkan->queue_image(s->images[id]);
        } catch (std::exception& e) { log_and_exit(e); }
        return 0;
}

constexpr std::array <codec_t, 1> codecs = { RGBA };

int display_sdl2_get_property(void* state, int property, void* val, size_t* len) {
        auto s = static_cast<state_vulkan_sdl2*>(state);
        assert(s->mod.priv_magic == MAGIC_VULKAN_SDL2);

        switch (property) {
        case DISPLAY_PROPERTY_CODECS: {
                size_t codecs_len = codecs.size() * sizeof(codec_t);
                if (codecs_len > *len) {
                        return FALSE;
                }
                memcpy(val, codecs.data(), codecs_len);
                *len = codecs_len;
                break;
        }
        case DISPLAY_PROPERTY_BUF_PITCH: {
                if (sizeof(int) > *len) {
                        return FALSE;
                }
                int* value = reinterpret_cast<int*>(val);
                *len = sizeof(int);

                vkd::image image;
                const auto& desc = s->current_desc;
                assert(s->current_desc.width != 0);
                s->vulkan->acquire_image(image, { desc.width, desc.height });
                *value = static_cast<int>(image.get_row_pitch());
                s->vulkan->discard_image(image);
                break;
        }
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

