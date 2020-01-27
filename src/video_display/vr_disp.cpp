
#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <assert.h>
#include <SDL2/SDL.h>

#include "debug.h"
#include "host.h"
#include "keyboard_control.h" // K_*
#include "lib_common.h"
#include "messaging.h"
#include "module.h"
#include "video.h"
#include "video_display.h"
#include "video_display/splashscreen.h"

struct Window{
	Window() : Window("UltraGrid VR",
			SDL_WINDOWPOS_CENTERED,
			SDL_WINDOWPOS_CENTERED,
			640,
			480,
			SDL_WINDOW_OPENGL) {  }

	Window(const char *title, int x, int y, int w, int h, SDL_WindowFlags flags){
		SDL_InitSubSystem(SDL_INIT_VIDEO);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);

		sdl_window = SDL_CreateWindow("UltraGrid VR",
				x,
				y,
				w,
				h,
				SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
	}


	~Window(){
		SDL_DestroyWindow(sdl_window);
		SDL_QuitSubSystem(SDL_INIT_VIDEO);
	}


	SDL_Window *sdl_window;
};

struct state_vr{
	Window window;
};

static void * display_vr_init(struct module *parent, const char *fmt, unsigned int flags) {
	state_vr *s = new state_vr();

	return s;
}

static void display_vr_run(void *arg) {

}

static void display_vr_done(void *state) {
	state_vr *s = static_cast<state_vr *>(state);

	delete s;
}

static struct video_frame * display_vr_getf(void *state) {

}

static int display_vr_putf(void *state, struct video_frame *frame, int nonblock) {

}

static int display_vr_reconfigure(void *state, struct video_desc desc) {

}

static int display_vr_get_property(void *state, int property, void *val, size_t *len) {

}

static void display_vr_put_audio_frame(void *state, struct audio_frame *frame) {

}

static int display_vr_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate) {

}

static const struct video_display_info display_vr_info = {
        [](struct device_info **available_cards, int *count, void (**deleter)(void *)) {
                UNUSED(deleter);
                *count = 1;
                *available_cards = (struct device_info *) calloc(1, sizeof(struct device_info));
                strcpy((*available_cards)[0].id, "vr");
                strcpy((*available_cards)[0].name, "VR SW display");
                (*available_cards)[0].repeatable = true;
        },
        display_vr_init,
        display_vr_run,
        display_vr_done,
        display_vr_getf,
        display_vr_putf,
        display_vr_reconfigure,
        display_vr_get_property,
        display_vr_put_audio_frame,
        display_vr_reconfigure_audio,
};

REGISTER_MODULE(vr_disp, &display_vr_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);
