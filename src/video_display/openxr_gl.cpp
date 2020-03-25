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

class Texture{
public:
	Texture(){
		glGenTextures(1, &tex_id);
		glBindTexture(GL_TEXTURE_2D, tex_id);
		//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 4, 4, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	~Texture(){
		glDeleteTextures(1, &tex_id);
	}

	GLuint get() const { return tex_id; }

	void allocate(int w, int h, GLenum fmt) {
		if(w != width || h != height || fmt != format){
			width = w;
			height = h;
			format = fmt;

			glBindTexture(GL_TEXTURE_2D, tex_id);
			glTexImage2D(GL_TEXTURE_2D, 0, format,
					width, height, 0,
					format, GL_UNSIGNED_BYTE,
					nullptr);
		}
	}

	Texture(const Texture&) = delete;
	Texture(Texture&& o) { swap(o); }
	Texture& operator=(const Texture&) = delete;
	Texture& operator=(Texture&& o) { swap(o); return *this; }

private:
	void swap(Texture& o){
		std::swap(tex_id, o.tex_id);
	}
	GLuint tex_id = 0;
	int width = 0;
	int height = 0;
	GLenum format = 0;
};

class Framebuffer{
public:
	Framebuffer(){
		glGenFramebuffers(1, &fbo);
	}

	~Framebuffer(){
		glDeleteFramebuffers(1, &fbo);
	}

	Framebuffer(const Framebuffer&) = delete;
	Framebuffer(Framebuffer&& o) { swap(o); }
	Framebuffer& operator=(const Framebuffer&) = delete;
	Framebuffer& operator=(Framebuffer&& o) { swap(o); return *this; }

	GLuint get() { return fbo; }

	void attach_texture(GLuint tex){
		glBindTexture(GL_TEXTURE_2D, tex);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glBindTexture(GL_TEXTURE_2D, 0);

		glBindFramebuffer(GL_FRAMEBUFFER, fbo);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
		assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	void attach_texture(const Texture& tex){
		attach_texture(tex.get());
	}


private:
	void swap(Framebuffer& o){
		std::swap(fbo, o.fbo);
	}

	GLuint fbo = 0;
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
		graphics_binding_gl.xDisplay = xDisplay;
		graphics_binding_gl.glxContext = glxContext;
		graphics_binding_gl.glxDrawable = glxDrawable;

		XrSessionCreateInfo session_create_info = {};
		session_create_info.type = XR_TYPE_SESSION_CREATE_INFO;
		session_create_info.next = &graphics_binding_gl;
		session_create_info.systemId = systemId;


		XrResult result = xrCreateSession(instance, &session_create_info, &session);
		//TODO Error check
	}

	~Openxr_session(){
		xrDestroySession(session);
	}

	XrSession get(){ return session; }

	void begin(){
		XrSessionBeginInfo session_begin_info;
		session_begin_info.type = XR_TYPE_SESSION_BEGIN_INFO;
		session_begin_info.next = NULL;
		session_begin_info.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
		XrResult result = xrBeginSession(session, &session_begin_info);
		printf("Session started!\n");
	}
private:
	XrSession session;
};

class Openxr_instance{
public:
	Openxr_instance(){
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

	~Openxr_instance(){
		xrDestroyInstance(instance);
	}

	Openxr_instance(const Openxr_instance&) = delete;
	Openxr_instance(Openxr_instance&&) = delete;
	Openxr_instance& operator=(const Openxr_instance&) = delete;
	Openxr_instance& operator=(Openxr_instance&&) = delete;

	XrInstance get() { return instance; }
private:
	XrInstance instance;
};

class Openxr_swapchain{
public:
	Openxr_swapchain(XrSession session, const XrSwapchainCreateInfo *info) :
		session(session)
	{
		xrCreateSwapchain(session, info, &swapchain);
	}

	Openxr_swapchain(XrSession session,
			int64_t swapchain_format,
			uint32_t w,
			uint32_t h) :
		session(session)
	{
		XrSwapchainCreateInfo swapchain_create_info;
		swapchain_create_info.type = XR_TYPE_SWAPCHAIN_CREATE_INFO;
		swapchain_create_info.usageFlags = XR_SWAPCHAIN_USAGE_SAMPLED_BIT |
			XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
		swapchain_create_info.createFlags = 0;
		swapchain_create_info.format = swapchain_format;
		swapchain_create_info.sampleCount = 1;
		swapchain_create_info.width = w;
		swapchain_create_info.height = h;
		swapchain_create_info.faceCount = 1;
		swapchain_create_info.arraySize = 1;
		swapchain_create_info.mipCount = 1;
		swapchain_create_info.next = nullptr;

		xrCreateSwapchain(session, &swapchain_create_info, &swapchain);
	}

	~Openxr_swapchain(){
		xrDestroySwapchain(swapchain);
	}

	Openxr_swapchain(const Openxr_swapchain&) = delete;
	Openxr_swapchain(Openxr_swapchain&& o) { swap(o); }
	Openxr_swapchain& operator=(const Openxr_swapchain&) = delete;
	Openxr_swapchain& operator=(Openxr_swapchain&& o) { swap(o); return *this; }

	uint32_t get_length(){
		uint32_t len = 0;
		//TODO error check
		xrEnumerateSwapchainImages(swapchain, 0, &len, nullptr);

		return len;
	}

	void swap(Openxr_swapchain& o){
		std::swap(swapchain, o.swapchain);
		std::swap(session, o.session);
	}

	XrSwapchain get(){ return swapchain; }

private:
	XrSwapchain swapchain;
	XrSession session;
};

class Gl_interop_swapchain{
public:
	Gl_interop_swapchain(XrSession session, const XrSwapchainCreateInfo *info) :
		xr_swapchain(session, info)
	{
		init();
	}

	Gl_interop_swapchain(XrSession session,
			int64_t swapchain_format,
			uint32_t w,
			uint32_t h) :
		xr_swapchain(session, swapchain_format, w, h)
	{
		init();
	}

	GLuint get_texture(size_t idx){
		return images[idx].image;
	}

	Framebuffer& get_framebuffer(size_t idx){
		return framebuffers[idx];
	}

	Gl_interop_swapchain(const Gl_interop_swapchain&) = delete;
	Gl_interop_swapchain(Gl_interop_swapchain&&) = default;
	Gl_interop_swapchain& operator=(const Gl_interop_swapchain&) = delete;
	Gl_interop_swapchain& operator=(Gl_interop_swapchain&&) = default;

private:
	void init(){
		uint32_t length = xr_swapchain.get_length();
		images.resize(length);
		framebuffers.resize(length);

		for(auto& image : images){
			image.type = XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_KHR;
			image.next = nullptr;
		}

		xrEnumerateSwapchainImages(xr_swapchain.get(),
				length, &length,
				(XrSwapchainImageBaseHeader *)(images.data()));

		for(size_t i = 0; i < length; i++){
			framebuffers[i].attach_texture(get_texture(i));
		}
	}

	Openxr_swapchain xr_swapchain;
	std::vector<Framebuffer> framebuffers;
	std::vector<XrSwapchainImageOpenGLKHR> images;
};

struct Openxr_state{
	Openxr_instance instance;
	XrSystemId system_id;
	//Openxr_session session;	
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

static std::vector<XrViewConfigurationView> get_views(Openxr_state& xr_state){
	unsigned view_count;
	XrResult result = xrEnumerateViewConfigurationViews(xr_state.instance.get(),
			xr_state.system_id,
			XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
			0,
			&view_count,
			nullptr);

	std::vector<XrViewConfigurationView> config_views(view_count);
	for(auto& view : config_views) view.type = XR_TYPE_VIEW_CONFIGURATION_VIEW;

	result = xrEnumerateViewConfigurationViews(xr_state.instance.get(),
			xr_state.system_id,
			XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
			view_count,
			&view_count,
			config_views.data());

	return config_views;
}

static std::vector<int64_t> get_swapchain_formats(XrSession session){
	XrResult result;
	unsigned swapchain_format_count;
	result = xrEnumerateSwapchainFormats(session,
			0,
			&swapchain_format_count,
			nullptr);

	printf("Runtime supports %d swapchain formats\n", swapchain_format_count);
	std::vector<int64_t> swapchain_formats(swapchain_format_count);
	result = xrEnumerateSwapchainFormats(session,
			swapchain_format_count,
			&swapchain_format_count,
			swapchain_formats.data());

	return swapchain_formats;
}

static void display_xrgl_run(void *state){
	state_xrgl *s = static_cast<state_xrgl *>(state);

	Display *xDisplay = nullptr;
	GLXContext glxContext;
	GLXDrawable glxDrawable;

	s->window.getXlibHandles(&xDisplay, &glxContext, &glxDrawable);

	Openxr_session session(s->xr_state.instance.get(),
			s->xr_state.system_id,
			xDisplay,
			glxContext,
			glxDrawable);

	std::vector<XrViewConfigurationView> config_views = get_views(s->xr_state);

	session.begin();

	std::vector<int64_t> swapchain_formats = get_swapchain_formats(session.get());
	int64_t selected_swapchain_format = swapchain_formats[0];

	std::vector<Gl_interop_swapchain> swapchains;
	for(const auto& view : config_views){
		swapchains.emplace_back(session.get(),
				selected_swapchain_format,
				view.recommendedImageRectWidth,
				view.recommendedImageRectHeight);
	}


}

static void * display_xrgl_init(struct module *parent, const char *fmt, unsigned int flags) {
	state_xrgl *s = new state_xrgl();

	s->xr_state.system_id = 1; //TODO

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

static void display_xrgl_probe(struct device_info **available_cards, int *count, void (**deleter)(void *)){
	UNUSED(deleter);
	*count = 0;
	*available_cards = nullptr;

	Openxr_instance instance;

	XrSystemGetInfo systemGetInfo;
	systemGetInfo.type = XR_TYPE_SYSTEM_GET_INFO;
	systemGetInfo.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
	systemGetInfo.next = NULL;

	XrSystemId systemId;
	XrResult result = xrGetSystem(instance.get(), &systemGetInfo, &systemId);
	if(!XR_SUCCEEDED(result)){
		return;
	}

	XrSystemProperties systemProperties;
	systemProperties.type = XR_TYPE_SYSTEM_PROPERTIES;
	systemProperties.next = NULL;
	systemProperties.graphicsProperties = {0};
	systemProperties.trackingProperties = {0};

	result = xrGetSystemProperties(instance.get(), systemId, &systemProperties);
	if(!XR_SUCCEEDED(result)){
		return;
	}

	*available_cards = (struct device_info *) calloc(1, sizeof(struct device_info));
	*count = 1;
	snprintf((*available_cards)[0].id, sizeof((*available_cards)[0].id), "openxr_gl:system=%lu", systemId);
	snprintf((*available_cards)[0].name, sizeof((*available_cards)[0].name), "OpenXr: %s", systemProperties.systemName);
	(*available_cards)[0].repeatable = false;
}


static const struct video_display_info openxr_gl_info = {
		display_xrgl_probe,
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
