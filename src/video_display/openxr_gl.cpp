#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <chrono>

#include <assert.h>
//#include <GL/glew.h>
#include <X11/Xlib.h>
#include <GL/glew.h>
#include <GL/glx.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#define XR_USE_PLATFORM_XLIB
#define XR_USE_GRAPHICS_API_OPENGL
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "debug.h"
#include "host.h"
#include "keyboard_control.h" // K_*
#include "lib_common.h"
#include "messaging.h"
#include "module.h"
#include "video.h"
#include "video_display.h"
#include "video_display/splashscreen.h"

#include "utils/profile_timer.hpp"

#define MAX_BUFFER_SIZE   1

static const float PI_F=3.14159265358979f;

struct Sdl_window{
	Sdl_window() : Sdl_window("UltraGrid VR",
			SDL_WINDOWPOS_CENTERED,
			SDL_WINDOWPOS_CENTERED,
			640,
			480,
			SDL_WINDOW_OPENGL) {  }

	Sdl_window(const char *title, int x, int y, int w, int h, SDL_WindowFlags flags) :
	width(w), height(h)
	{
		SDL_InitSubSystem(SDL_INIT_VIDEO);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);

		SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 0);

		//TODO: Error handling
		sdl_window = SDL_CreateWindow(title,
				x,
				y,
				w,
				h,
				SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);

		SDL_SetWindowMinimumSize(sdl_window, 200, 200);

		//TODO: Error handling
		sdl_gl_context = SDL_GL_CreateContext(sdl_window);

		glewExperimental = GL_TRUE;
		//TODO: Error handling
		GLenum glewError = glewInit();

		glClearColor(0,0,0,1);
		glClear(GL_COLOR_BUFFER_BIT);
		SDL_GL_SwapWindow(sdl_window);
	}


	~Sdl_window(){
		SDL_DestroyWindow(sdl_window);
		SDL_QuitSubSystem(SDL_INIT_VIDEO);
	}

	void getXlibHandles(Display  **xDisplay,
			GLXContext *glxContext,
			GLXDrawable *glxDrawable)
	{
		SDL_GL_MakeCurrent(sdl_window, sdl_gl_context);
		*xDisplay = XOpenDisplay(NULL);
		*glxContext = glXGetCurrentContext();
		*glxDrawable = glXGetCurrentDrawable();
	}


	SDL_Window *sdl_window;
	SDL_GLContext sdl_gl_context;
	int width;
	int height;
};

class Openxr_session{
public:
	Openxr_session(XrInstance instance,
			XrSystemId systemId,
			Display *xDisplay,
			GLXContext glxContext,
			GLXDrawable glxDrawable)
	{
		XrGraphicsBindingOpenGLXlibKHR graphics_binding_gl = {};
	    graphics_binding_gl.type = XR_TYPE_GRAPHICS_BINDING_OPENGL_XLIB_KHR;

		XrSessionCreateInfo session_create_info = {};
		session_create_info.type = XR_TYPE_SESSION_CREATE_INFO;
		session_create_info.next = &self->graphics_binding_gl;
		session_create_info.systemId = systemId;


		XrResult result = xrCreateSession(instance, &session_create_info, &session);
		//TODO Error check
	}
private:
	XrSession session;
};

class Openxr_state{
public:
	Openxr_state(){
		//TODO Check if opengl extension is supported

		const char* const enabledExtensions[] = {XR_KHR_OPENGL_ENABLE_EXTENSION_NAME};

		XrInstanceCreateInfo instanceCreateInfo;
		instanceCreateInfo.type = XR_TYPE_INSTANCE_CREATE_INFO;
		instanceCreateInfo.next = NULL;
		instanceCreateInfo.createFlags = 0;
		instanceCreateInfo.enabledExtensionCount = 1;
		instanceCreateInfo.enabledExtensionNames = enabledExtensions;
		instanceCreateInfo.enabledApiLayerCount = 0;
		strcpy(instanceCreateInfo.applicationInfo.applicationName, "UltraGrid OpenXR gl display");
		strcpy(instanceCreateInfo.applicationInfo.engineName, "");
		instanceCreateInfo.applicationInfo.applicationVersion = 1;
		instanceCreateInfo.applicationInfo.engineVersion = 0;
		instanceCreateInfo.applicationInfo.apiVersion = XR_CURRENT_API_VERSION;

		XrResult result = xrCreateInstance(&instanceCreateInfo, &instance);

		XrInstanceProperties instanceProperties;
		instanceProperties.type = XR_TYPE_INSTANCE_PROPERTIES;
		instanceProperties.next = NULL;

		result = xrGetInstanceProperties(instance, &instanceProperties);

		printf("Runtime Name: %s\n", instanceProperties.runtimeName);
		printf("Runtime Version: %d.%d.%d\n",
		       XR_VERSION_MAJOR(instanceProperties.runtimeVersion),
		       XR_VERSION_MINOR(instanceProperties.runtimeVersion),
		       XR_VERSION_PATCH(instanceProperties.runtimeVersion));


	}

	~Openxr_state(){
		xrDestroyInstance(instance);
	}

	Openxr_state(const Openxr_state&) = delete;
	Openxr_state(Openxr_state&&) = delete;
	Openxr_state& operator=(const Openxr_state&) = delete;
	Openxr_state& operator=(Openxr_state&&) = delete;
private:
	XrInstance instance;
	XrSession session;
};

struct state_xrgl{
	video_desc current_desc;
	int buffered_frames_count;

	Sdl_window window;
	Openxr_state xr_state;

	std::chrono::steady_clock::time_point last_frame;

	std::mutex lock;
	std::condition_variable frame_consumed_cv;
	std::queue<video_frame *> free_frame_queue;
};

static void display_xrgl_run(void *state){
	state_xrgl *s = static_cast<state_xrgl *>(state);
}

static void * display_xrgl_init(struct module *parent, const char *fmt, unsigned int flags) {
	state_xrgl *s = new state_xrgl();

	return s;
}

static void display_xrgl_done(void *state) {
	state_xrgl *s = static_cast<state_xrgl *>(state);

	delete s;
}

static struct video_frame * display_xrgl_getf(void *state) {
	struct state_xrgl *s = static_cast<state_xrgl *>(state);

	std::lock_guard<std::mutex> lock(s->lock);

	while (s->free_frame_queue.size() > 0) {
		struct video_frame *buffer = s->free_frame_queue.front();
		s->free_frame_queue.pop();
		if (video_desc_eq(video_desc_from_frame(buffer), s->current_desc)) {
			return buffer;
		} else {
			vf_free(buffer);
		}
	}

	return vf_alloc_desc_data(s->current_desc);
}

static int display_xrgl_putf(void *state, struct video_frame *frame, int nonblock) {
	struct state_xrgl *s = static_cast<state_xrgl *>(state);

	if (nonblock == PUTF_DISCARD) {
		vf_free(frame);
		return 0;
	}

	std::unique_lock<std::mutex> lk(s->lock);
	if (s->buffered_frames_count >= MAX_BUFFER_SIZE && nonblock == PUTF_NONBLOCK
			&& frame != NULL) {
		vf_free(frame);
		printf("1 frame(s) dropped!\n");
		return 1;
	}
	s->frame_consumed_cv.wait(lk, [s]{return s->buffered_frames_count < MAX_BUFFER_SIZE;});
	s->buffered_frames_count += 1;
	lk.unlock();

	return 0;
}

static int display_xrgl_reconfigure(void *state, struct video_desc desc) {
	state_xrgl *s = static_cast<state_xrgl *>(state);

	s->current_desc = desc;
	return 1;
}

static int display_xrgl_get_property(void *state, int property, void *val, size_t *len) {
	UNUSED(state);
	codec_t codecs[] = {
		RGBA,
		RGB,
	};
	enum interlacing_t supported_il_modes[] = {PROGRESSIVE};
	int rgb_shift[] = {0, 8, 16};

	switch (property) {
		case DISPLAY_PROPERTY_CODECS:
			if(sizeof(codecs) <= *len) {
				memcpy(val, codecs, sizeof(codecs));
			} else {
				return FALSE;
			}

			*len = sizeof(codecs);
			break;
		case DISPLAY_PROPERTY_RGB_SHIFT:
			if(sizeof(rgb_shift) > *len) {
				return FALSE;
			}
			memcpy(val, rgb_shift, sizeof(rgb_shift));
			*len = sizeof(rgb_shift);
			break;
		case DISPLAY_PROPERTY_BUF_PITCH:
			*(int *) val = PITCH_DEFAULT;
			*len = sizeof(int);
			break;
		case DISPLAY_PROPERTY_SUPPORTED_IL_MODES:
			if(sizeof(supported_il_modes) <= *len) {
				memcpy(val, supported_il_modes, sizeof(supported_il_modes));
			} else {
				return FALSE;
			}
			*len = sizeof(supported_il_modes);
			break;
		default:
			return FALSE;
	}
	return TRUE;
}


static const struct video_display_info openxr_gl_info = {
        [](struct device_info **available_cards, int *count, void (**deleter)(void *)) {
                UNUSED(deleter);
                *count = 1;
                *available_cards = (struct device_info *) calloc(1, sizeof(struct device_info));
				//TODO: hmd querying
                strcpy((*available_cards)[0].id, "xrgl");
                strcpy((*available_cards)[0].name, "OpenXR gl display");
                (*available_cards)[0].repeatable = true;
        },
        display_xrgl_init,
        display_xrgl_run,
        display_xrgl_done,
        display_xrgl_getf,
        display_xrgl_putf,
        display_xrgl_reconfigure,
        display_xrgl_get_property,
        NULL,
        NULL,
};

REGISTER_MODULE(openxr_gl, &openxr_gl_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);
