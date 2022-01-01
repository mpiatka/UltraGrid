/**
 * @file   video_display/vulkan_sdl2.cpp
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
#include "utils/fs.h"
//remove leaking macros
#undef min
#undef max

#ifdef __MINGW32__
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#include "vulkan_display.h" // vulkan.h must be before GLFW/SDL

// @todo remove the defines when no longer needed
#ifdef __arm64__
#define SDL_DISABLE_MMINTRIN_H 1
#define SDL_DISABLE_IMMINTRIN_H 1
#endif // defined __arm64__

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

#ifdef __MINGW32__
#include <SDL2/SDL_syswm.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <charconv>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <queue>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility> // pair

using rang::fg;
using rang::style;

namespace vkd = vulkan_display;
namespace chrono = std::chrono;
using namespace std::literals;

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
        explicit window_callback(SDL_Window* window):
                window{window} { }
        
        vkd::window_parameters get_window_parameters() override {
                assert(window);
                int width = 0;
                int height = 0;
                SDL_Vulkan_GetDrawableSize(window, &width, &height);
                if (SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED) {
                        width = 0;
                        height = 0;
                }
                return { static_cast<uint32_t>(width), static_cast<uint32_t>(height), true };
        }
};

struct state_vulkan_sdl2 {
        module mod{};

        Uint32 sdl_user_new_message_event;

        chrono::steady_clock::time_point time{};
        uint64_t frames = 0;

        bool deinterlace = false;
        bool fullscreen = false;
        bool keep_aspect = false;
       
        int width = 0;
        int height = 0;
        
        SDL_Window* window = nullptr;
        std::unique_ptr<vkd::vulkan_display> vulkan = nullptr;
        std::unique_ptr<::window_callback> window_callback = nullptr;
        
        std::array<video_frame, MAX_FRAME_COUNT> video_frames{};
        std::array<vkd::image, MAX_FRAME_COUNT> images{};

        std::atomic<bool> should_exit = false;
        video_desc current_desc{};

        explicit state_vulkan_sdl2(module* parent) {
                module_init_default(&mod);
                mod.priv_magic = MAGIC_VULKAN_SDL2;
                mod.new_message = display_sdl2_new_message;
                mod.cls = MODULE_CLASS_DATA;
                module_register(&mod, parent);

                sdl_user_new_message_event = SDL_RegisterEvents(1);
                assert(sdl_user_new_message_event != static_cast<Uint32>(-1));
        }

        state_vulkan_sdl2(const state_vulkan_sdl2& other) = delete;
        state_vulkan_sdl2& operator=(const state_vulkan_sdl2& other) = delete;
        state_vulkan_sdl2(state_vulkan_sdl2&& other) = delete;
        state_vulkan_sdl2& operator=(state_vulkan_sdl2&& other) = delete;

        ~state_vulkan_sdl2() {
                module_done(&mod);
        }
};

// make sure that state_vulkan_sdl2 is C compatible
static_assert(std::is_standard_layout_v<state_vulkan_sdl2>);

//todo C++20 : change to to_array
constexpr std::array<std::pair<char, std::string_view>, 3> display_sdl2_keybindings{{
        {'d', "toggle deinterlace"},
        {'f', "toggle fullscreen"},
        {'q', "quit"}
}};

constexpr void update_description(const video_desc& video_desc, video_frame& frame) {
        frame.color_spec = video_desc.color_spec;
        frame.fps = video_desc.fps;
        frame.interlacing = video_desc.interlacing;
        frame.tile_count = video_desc.tile_count;
        for (unsigned i = 0; i < video_desc.tile_count; i++) {
                frame.tiles[i].width = video_desc.width;
                frame.tiles[i].height = video_desc.height;
        }
}

constexpr int64_t translate_sdl_key_to_ug(SDL_Keysym sym) {
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

constexpr bool display_sdl2_process_key(state_vulkan_sdl2& s, int64_t key) {
        switch (key) {
        case 'd':
                s.deinterlace = !s.deinterlace;
                log_msg(LOG_LEVEL_INFO, "Deinterlacing: %s\n",
                        s.deinterlace ? "ON" : "OFF");
                return true;
        case 'f': {
                s.fullscreen = !s.fullscreen;
                int mouse_x = 0, mouse_y = 0;
                SDL_GetGlobalMouseState(&mouse_x, &mouse_y);
                SDL_SetWindowFullscreen(s.window, s.fullscreen ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0);
                SDL_WarpMouseGlobal(mouse_x, mouse_y);
                return true;
        }
        case 'q':
                exit_uv(0);
                return true;
        default:
                return false;
        }
}

void log_and_exit_uv(std::exception& e) {
        LOG(LOG_LEVEL_ERROR) << MOD_NAME << e.what() << std::endl;
        exit_uv(EXIT_FAILURE);
}

void process_user_messages(state_vulkan_sdl2& s) {
        msg_universal* msg = nullptr;
        while ((msg = reinterpret_cast<msg_universal*>(check_message(&s.mod)))) {
                log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Received message: %s\n", msg->text);
                response* r = nullptr;
                int key;
                if (strstr(msg->text, "win-title ") == msg->text) {
                        SDL_SetWindowTitle(s.window, msg->text + strlen("win-title "));
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
}

void process_events(state_vulkan_sdl2& s) {
        SDL_Event sdl_event;
        while (SDL_PollEvent(&sdl_event)) {
                if (sdl_event.type == s.sdl_user_new_message_event) {
                        process_user_messages(s);
                } else if (sdl_event.type == SDL_KEYDOWN && sdl_event.key.repeat == 0) {
                        auto& keysym = sdl_event.key.keysym;
                        log_msg(LOG_LEVEL_VERBOSE, 
                                MOD_NAME "Pressed key %s (scancode: %d, sym: %d, mod: %d)!\n",
                                SDL_GetKeyName(keysym.sym), 
                                keysym.scancode, 
                                keysym.sym, 
                                keysym.mod);
                        int64_t sym = translate_sdl_key_to_ug(keysym);
                        if (sym > 0) {
                                if (!display_sdl2_process_key(s, sym)) { 
                                        // unknown key -> pass to control
                                        keycontrol_send_key(get_root_module(&s.mod), sym);
                                }
                        } else if (sym == -1) {
                                log_msg(LOG_LEVEL_WARNING, 
                                        MOD_NAME "Cannot translate key %s (scancode: %d, sym: %d, mod: %d)!\n",
                                        SDL_GetKeyName(keysym.sym), 
                                        keysym.scancode, 
                                        keysym.sym, 
                                        keysym.mod);
                        }

                } else if (sdl_event.type == SDL_WINDOWEVENT) {
                        // https://forums.libsdl.org/viewtopic.php?p=38342
                        if (s.keep_aspect && sdl_event.window.event == SDL_WINDOWEVENT_RESIZED) {
                                int old_width = s.width, old_height = s.height;
                                auto width = sdl_event.window.data1;
                                auto height = sdl_event.window.data2;
                                if (old_width == width) {
                                        width = old_width * height / old_height;
                                } else if (old_height == height) {
                                        height = old_height * width / old_width;
                                } else {
                                        auto area = int64_t{ width } * height;
                                        width = sqrt(area * old_width / old_height);
                                        height = sqrt(area * old_height / old_width);
                                }
                                s.width = width;
                                s.height = height;
                                SDL_SetWindowSize(s.window, width, height);
                                debug_msg(MOD_NAME "resizing to %d x %d\n", width, height);
                        } else if (sdl_event.window.event == SDL_WINDOWEVENT_EXPOSED
                                || sdl_event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED)
                        {
                                s.vulkan->window_parameters_changed();
                        }

                } else if (sdl_event.type == SDL_QUIT) {
                        exit_uv(0);
                }
        }
}

void display_sdl2_run(void* state) {
        auto* s = static_cast<state_vulkan_sdl2*>(state);
        assert(s->mod.priv_magic == MAGIC_VULKAN_SDL2);
        
        s->time = chrono::steady_clock::now();
        while (!s->should_exit) {
                process_events(*s);
                bool displayed = false;
                try {
                        s->vulkan->display_queued_image(&displayed);
                } 
                catch (std::exception& e) { log_and_exit_uv(e); break; }
                if (displayed) {
                        s->frames++;
                }
                auto now = chrono::steady_clock::now();
                double seconds = chrono::duration<double>{ now - s->time }.count();
                if (seconds > 5) {
                        double fps = s->frames / seconds;
                        log_msg(LOG_LEVEL_INFO, MOD_NAME "%llu frames in %g seconds = %g FPS\n",
                                static_cast<long long unsigned>(s->frames), seconds, fps);
                        s->time = now;
                        s->frames = 0;
                }
        }
        SDL_HideWindow(s->window);

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
        vkd::vulkan_instance instance;
        std::vector<const char*>required_extensions{};
        std::vector<std::pair<std::string, bool>> gpus{};
        try {
                instance.init(required_extensions, false);
                instance.get_available_gpus(gpus);
        } 
        catch (std::exception& e) { log_and_exit_uv(e); return; }
        
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
        using std::cout;

        auto print_drivers = []() {
                for (int i = 0; i < SDL_GetNumVideoDrivers(); ++i) {
                        std::cout << (i == 0 ? "" : ", ") << style::bold << SDL_GetVideoDriver(i) << style::reset;
                }
                std::cout << '\n';
        };
        
        cout << "VULKAN_SDL2 options:\n";
        cout << style::bold << fg::red << "\t-d vulkan_sdl2" << fg::reset;
        cout << "[:d|:fs|:keep-aspect|:nocursor|:nodecorate|:novsync|:validation|:display=<dis_id>|"
                ":driver=<drv>|:gpu=<gpu_id>|:pos=<x>,<y>|:size=<W>x<H>|:window_flags=<f>|:help]\n";

        cout << style::reset << ("\twhere:\n");

        cout << style::bold << "\t\t       d" << style::reset << " - deinterlace\n";
        cout << style::bold << "\t\t      fs" << style::reset << " - fullscreen\n";
        
        cout << style::bold << "\t     keep-aspect" << style::reset << " - keep window aspect ratio respecive to the video\n";
        cout << style::bold << "\t        nocursor" << style::reset << " - hides cursor\n";
        cout << style::bold << "\t      nodecorate" << style::reset << " - disable window border\n";
        cout << style::bold << "\t         novsync" << style::reset << " - disable vsync\n";
        cout << style::bold << "\t      validation" << style::reset << " - enable vulkan validation layers\n";

        cout << style::bold << "\tdisplay=<dis_id>" << style::reset << " - display index, available indices: ";
        sdl2_print_displays();
        cout << style::bold << "\t    driver=<drv>" << style::reset << " - available drivers:";
        print_drivers();
        cout << style::bold << "\t    gpu=<gpu_id>" << style::reset << " - gpu index selected from the following list\n";
        cout << style::bold << "\t     pos=<x>,<y>" << style::reset << " - set window position\n";
        cout << style::bold << "\t    size=<W>x<H>" << style::reset << " - set window size\n";
        cout << style::bold << "\twindow_flags=<f>" << style::reset << " - flags to be passed to SDL_CreateWindow (use prefix 0x for hex)\n";
        
        print_gpus();

        cout << "\n\tKeyboard shortcuts:\n";
        for (auto& binding : display_sdl2_keybindings) {
                cout << style::bold << "\t\t'" << binding.first << 
                        style::reset << "'\t - " << binding.second << "\n";
        }
        SDL_Quit();
}

int display_sdl2_reconfigure(void* state, video_desc desc) {
        auto* s = static_cast<state_vulkan_sdl2*>(state);
        assert(s->mod.priv_magic == MAGIC_VULKAN_SDL2);

        assert(desc.tile_count == 1);
        s->current_desc = desc;
        return TRUE;
}

/**
 * Load splashscreen
 * Function loads graphic data from header file "splashscreen.h", where are
 * stored splashscreen data in RGB format.
 */
void draw_splashscreen(state_vulkan_sdl2& s) {
        vkd::image image;
        try {
                s.vulkan->acquire_image(image, {splash_width, splash_height, vk::Format::eR8G8B8A8Srgb});
        } 
        catch (std::exception& e) { log_and_exit_uv(e); return; }
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
        try {
                s.vulkan->queue_image(image);
                s.vulkan->display_queued_image();
        } 
        catch (std::exception& e) { log_and_exit_uv(e); }
}

// todo C++20: replace with member function
constexpr bool starts_with(std::string_view str, std::string_view match){
        return str.rfind(match, /*check only 0-th pos*/ 0) == 0;
};

struct command_line_arguments {
        bool cursor = true;
        bool help = false;
        bool vsync = true;
        bool validation = false;

        int display_idx = 0;
        int x = SDL_WINDOWPOS_UNDEFINED;
        int y = SDL_WINDOWPOS_UNDEFINED;

        uint32_t window_flags = 0 ; ///< user requested flags
        uint32_t gpu_idx = vkd::NO_GPU_SELECTED;
        std::string driver{};
};

bool parse_command_line_arguments(command_line_arguments& args, state_vulkan_sdl2& s, std::string_view arguments_sv) {
        constexpr auto npos = std::string_view::npos;
        constexpr std::string_view wrong_option_msg = MOD_NAME "Wrong option: ";

        // todo C++20: replace with std::views::split(options, ":")
        auto next_token = [](std::string_view & options) -> std::string_view {
                auto colon_pos = options.find(':');
                auto token = options.substr(0, colon_pos);
                options.remove_prefix(colon_pos == npos ? options.size() : colon_pos + 1);
                return token;
        };
        
        auto& res = args;
        while (!arguments_sv.empty()) try {
                const std::string_view token = next_token(arguments_sv);
                if (token.empty()) {
                        continue;
                }

                //svtoi = string_view to int
                auto svtoi = [token](std::string_view str) -> int {
                        int base = 10;
                        if (starts_with(str, "0x")) {
                                base = 16;
                                str.remove_prefix("0x"sv.size());
                        }
                        const char* last = str.data() + str.size();
                        int result = 0;
                        auto [ptr, err] = std::from_chars(str.data(), last, result, base);
                        constexpr auto no_error = std::errc{};
                        if (err != no_error || ptr != last) {
                                throw std::runtime_error{ std::string(token) };
                        }
                        return result;
                };

                if (token == "d") {
                        s.deinterlace = true;
                } else if (token == "fs") {
                        s.fullscreen = true;
                } else if (token == "keep-aspect") {
                        s.keep_aspect = true;
                } else if (token == "nocursor") {
                        args.cursor = false;
                } else if (token == "nodecorate") {
                        args.window_flags |= SDL_WINDOW_BORDERLESS;
                } else if (token == "novsync") {
                        res.vsync = false;
                } else if (token == "validation") {
                        res.validation = true;
                } else if (starts_with(token, "display=")) {
                        constexpr auto pos = "display="sv.size();
                        res.display_idx = svtoi(token.substr(pos));
                } else if (starts_with(token, "driver=")) {
                        constexpr auto pos = "driver="sv.size();
                        res.driver = std::string{ token.substr(pos) };
                } else if (starts_with(token, "gpu=")) {
                        constexpr auto pos = "gpu="sv.size();
                        args.gpu_idx = svtoi(token.substr(pos));
                } else if (starts_with(token, "pos=")) {
                        auto tok = token;
                        tok.remove_prefix("pos="sv.size());
                        auto comma = tok.find(',');
                        if (comma == npos) {
                                LOG(LOG_LEVEL_ERROR) << MOD_NAME "Missing colon in option:" 
                                        << token << '\n';
                                return false;
                        }
                        res.x = svtoi(tok.substr(0, comma));
                        res.y = svtoi(tok.substr(comma + 1));
                } else if (starts_with(token, "size=")) {
                        auto tok = token;
                        tok.remove_prefix("size="sv.size());
                        auto x = tok.find('x');
                        if (x == npos) {
                                LOG(LOG_LEVEL_ERROR) << MOD_NAME "Missing deliminer 'x' in option:" 
                                        << token << '\n';
                                return false;
                        }
                        s.width = svtoi(tok.substr(0, x));
                        s.height = svtoi(tok.substr(x + 1));
                } else if (starts_with(token, "window_flags=")) {
                        constexpr auto pos = "window_flags="sv.size();
                        int flags = svtoi(token.substr(pos));
                        args.window_flags |= flags;
                } else if (token == "help") {
                        show_help();
                        res.help = true;
                } else {
                        LOG(LOG_LEVEL_ERROR) << wrong_option_msg << token << '\n';
                        return false;
                }
        }
        catch (std::exception& e) {
                LOG(LOG_LEVEL_ERROR) << wrong_option_msg << e.what() << '\n';
                return false;
        }
        return true;
}

void* display_sdl2_init(module* parent, const char* fmt, unsigned int flags) {
        if (flags & DISPLAY_FLAG_AUDIO_ANY) {
                log_msg(LOG_LEVEL_ERROR, "UltraGrid VULKAN_SDL2 module currently doesn't support audio!\n");
                return nullptr;
        }

        auto s = std::make_unique<state_vulkan_sdl2>(parent);

        command_line_arguments args{};
        if (fmt) {
                if (!parse_command_line_arguments(args, *s, fmt)) {
                        return nullptr;
                }
                if (args.help) {
                        return &display_init_noerr;
                }
        }

        int ret = SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS);
        if (ret < 0) {
                log_msg(LOG_LEVEL_ERROR, "Unable to initialize SDL2: %s\n", SDL_GetError());
                return nullptr;
        }

        ret = SDL_VideoInit(args.driver.empty() ? nullptr : args.driver.c_str());
        if (ret < 0) {
                log_msg(LOG_LEVEL_ERROR, "Unable to initialize SDL2 video: %s\n", SDL_GetError());
                return nullptr;
        }
        log_msg(LOG_LEVEL_NOTICE, "[SDL] Using driver: %s\n", SDL_GetCurrentVideoDriver());

        SDL_ShowCursor(args.cursor);
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

        int x = (args.x == SDL_WINDOWPOS_UNDEFINED ? SDL_WINDOWPOS_CENTERED_DISPLAY(args.display_idx) : args.x);
        int y = (args.y == SDL_WINDOWPOS_UNDEFINED ? SDL_WINDOWPOS_CENTERED_DISPLAY(args.display_idx) : args.y);
        if (s->width == 0) s->width = 960;
        if (s->height == 0) s->height = 540;

        int window_flags = args.window_flags | SDL_WINDOW_RESIZABLE | SDL_WINDOW_VULKAN;
        if (s->fullscreen) {
                window_flags |= SDL_WINDOW_FULLSCREEN_DESKTOP;
        }

        s->window = SDL_CreateWindow(window_title, x, y, s->width, s->height, window_flags);
        if (!s->window) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to create window : %s\n", SDL_GetError());
                return nullptr;
        }
        s->window_callback = std::make_unique<::window_callback>(s->window);

        uint32_t extension_count = 0;
        SDL_Vulkan_GetInstanceExtensions(s->window, &extension_count, nullptr);
        std::vector<const char*> required_extensions(extension_count);
        SDL_Vulkan_GetInstanceExtensions(s->window, &extension_count, required_extensions.data());
        assert(extension_count > 0);
        
        std::string path = get_executable_path();
        std::filesystem::path path_to_shaders = {path.empty() ? "." : std::move(path)};
        path_to_shaders.remove_filename(); //remove uv or uv.exe
        path_to_shaders = path_to_shaders / "../share/ultragrid/vulkan_shaders";
        LOG(LOG_LEVEL_INFO) << MOD_NAME "Path to shaders: " << path_to_shaders << '\n';
        try {
                vkd::vulkan_instance instance;
                auto logging_function =
                        [](std::string_view sv) { LOG(LOG_LEVEL_INFO) << MOD_NAME << sv << std::endl; };
                instance.init(required_extensions, args.validation, logging_function);
#ifdef __MINGW32__
                //SDL2 for MINGW has problem creating surface
                SDL_SysWMinfo wmInfo{};
                SDL_VERSION(&wmInfo.version);
                SDL_GetWindowWMInfo(s->window, &wmInfo);
                HWND hwnd = wmInfo.info.win.window;
                HINSTANCE hinst = wmInfo.info.win.hinstance;
                vk::Win32SurfaceCreateInfoKHR create_info{};
                create_info
                        .setHinstance(hinst)
                        .setHwnd(hwnd);
                vk::SurfaceKHR surface = nullptr;
                if (instance.get_instance().createWin32SurfaceKHR(&create_info, nullptr, &surface) != vk::Result::eSuccess) {
                        throw std::runtime_error("Surface cannot be created.");
                }
#else
                VkSurfaceKHR surface = nullptr;
                if (!SDL_Vulkan_CreateSurface(s->window, instance.get_instance(), &surface)) {
                        std::cout << SDL_GetError() << std::endl;
                        throw std::runtime_error("SDL cannot create surface.");
                }
#endif
                s->vulkan = std::make_unique<vkd::vulkan_display>();
                s->vulkan->init(std::move(instance), surface, MAX_FRAME_COUNT, *s->window_callback, args.gpu_idx, path_to_shaders);
                LOG(LOG_LEVEL_NOTICE) << MOD_NAME "Vulkan display initialised." << std::endl;
        }
        catch (std::exception& e) { log_and_exit_uv(e); return nullptr; }

        for (auto& frame : s->video_frames) {
                frame = video_frame{};
        }

        draw_splashscreen(*s);
        return static_cast<void*>(s.release());
}

void display_sdl2_done(void* state) {
        auto* s = static_cast<state_vulkan_sdl2*>(state);
        assert(s->mod.priv_magic == MAGIC_VULKAN_SDL2);

        SDL_ShowCursor(SDL_ENABLE);

        try {
                s->vulkan->destroy();
        }
        catch (std::exception& e) { log_and_exit_uv(e); }
        s->vulkan = nullptr;

        if (s->window) {
                SDL_DestroyWindow(s->window);
                s->window = nullptr;
        }

        SDL_QuitSubSystem(SDL_INIT_EVENTS);
        SDL_Quit();

        delete s;
}

constexpr std::array<std::pair<codec_t, vk::Format>, 5> codec_to_vulkan_format_mapping {{
        {RGBA, vk::Format::eR8G8B8A8Srgb},
        {UYVY, vk::Format::eB8G8R8G8422Unorm},
        {YUYV, vk::Format::eG8B8G8R8422Unorm},
        {Y216, vk::Format::eG16B16G16R16422Unorm},
        {DXT1, vk::Format::eBc1RgbSrgbBlock}
}};

vkd::image_description to_vkd_image_desc(const video_desc& ultragrid_desc) {
        auto& mapping = codec_to_vulkan_format_mapping;
        codec_t searched_codec = ultragrid_desc.color_spec;
        auto iter = std::find_if(mapping.begin(), mapping.end(),
                [searched_codec](auto pair) { return pair.first == searched_codec; });
        vk::Format image_format = (iter == mapping.end()) ? vk::Format::eUndefined : iter->second;
        return { ultragrid_desc.width, ultragrid_desc.height, image_format };
}

video_frame* display_sdl2_getf(void* state) {
        auto* s = static_cast<state_vulkan_sdl2*>(state);
        assert(s->mod.priv_magic == MAGIC_VULKAN_SDL2);
        
        const auto& desc = s->current_desc;
        vulkan_display::image image;
        try {
                s->vulkan->acquire_image(image, to_vkd_image_desc(desc));
        } 
        catch (std::exception& e) { log_and_exit_uv(e); return nullptr; }
        s->images[image.get_id()] = image;
        
        video_frame& frame = s->video_frames[image.get_id()];
        update_description(desc, frame);
        auto texel_height = image.get_size().height;
        if (vkd::is_compressed_format(image.get_description().format)){
                texel_height /= 4;
        }
        frame.tiles[0].data_len = image.get_row_pitch() * texel_height;
        frame.tiles[0].data = reinterpret_cast<char*>(image.get_memory_ptr());
        return &frame;
}

int display_sdl2_putf(void* state, video_frame* frame, int nonblock) {
        auto* s = static_cast<state_vulkan_sdl2*>(state);
        assert(s->mod.priv_magic == MAGIC_VULKAN_SDL2);

        uint32_t id = std::distance(s->video_frames.data(), frame);
        auto& image = s->images[id];
        if (nonblock == PUTF_DISCARD) {
                assert(frame != nullptr);
                assert(image.get_id() == id);
                try {
                        s->vulkan->discard_image(s->images[id]);
                } 
                catch (std::exception& e) { log_and_exit_uv(e); return 1; }
                return 0;
        }

        if (!frame) {
                s->should_exit = true;
                s->vulkan->queue_image(vkd::image{});
                return 0;
        }
        assert(image.get_id() == id);
        
        if (s->deinterlace && !vkd::is_compressed_format(image.get_description().format)) {
                image.set_process_function([](vkd::image& image) {
                        vc_deinterlace(reinterpret_cast<unsigned char*>(image.get_memory_ptr()),
                                image.get_row_pitch(),
                                image.get_size().height);
                });
        }
        
        try {
                s->vulkan->queue_image(s->images[id]);
        } 
        catch (std::exception& e) { log_and_exit_uv(e); return 1; }
        return 0;
}

int display_sdl2_get_property(void* state, int property, void* val, size_t* len) {
        auto* s = static_cast<state_vulkan_sdl2*>(state);
        assert(s->mod.priv_magic == MAGIC_VULKAN_SDL2);

        switch (property) {
        case DISPLAY_PROPERTY_CODECS: {
                auto& mapping = codec_to_vulkan_format_mapping;
                std::vector<codec_t> codecs{};
                codecs.reserve(mapping.size());
                for (auto& pair : mapping) {
                        bool format_supported = false;
                        try {
                                s->vulkan->is_image_description_supported(format_supported,
                                        {1920, 1080, pair.second });
                        }
                        catch (std::exception& e) { log_and_exit_uv(e); return FALSE; }
                        if (format_supported) {
                                codecs.push_back(pair.first);
                        }
                }
                LOG(LOG_LEVEL_INFO) << MOD_NAME "Supported codecs are: ";
                for (auto codec : codecs) {
                        LOG(LOG_LEVEL_INFO) << get_codec_name(codec) << " ";
                }
                LOG(LOG_LEVEL_INFO) << std::endl;
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
                *len = sizeof(int);

                vkd::image image;
                const auto& desc = s->current_desc;
                assert(s->current_desc.width != 0);
                try {
                        s->vulkan->acquire_image(image, to_vkd_image_desc(desc));
                        auto value = static_cast<int>(image.get_row_pitch());
                        if (vkd::is_compressed_format(image.get_description().format)){
                                value /= 4;
                        }
                        memcpy(val, &value, sizeof(value));
                        s->vulkan->discard_image(image);
                } 
                catch (std::exception& e) { log_and_exit_uv(e); return FALSE; }
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

