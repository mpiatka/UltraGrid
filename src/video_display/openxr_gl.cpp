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
#include <glm/gtc/quaternion.hpp>

#include "debug.h"
#include "host.h"
#include "keyboard_control.h" // K_*
#include "lib_common.h"
#include "messaging.h"
#include "module.h"
#include "video.h"
#include "video_display.h"
#include "video_display/splashscreen.h"

#include "opengl_utils.hpp"
#include "utils/profile_timer.hpp"

#define MAX_BUFFER_SIZE   1

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
	Openxr_swapchain() = default;

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
		if(swapchain != XR_NULL_HANDLE){
			xrDestroySwapchain(swapchain);
		}
	}

	Openxr_swapchain(const Openxr_swapchain&) = delete;
	Openxr_swapchain(Openxr_swapchain&& o) noexcept : Openxr_swapchain() {
		swap(o);
	}
	Openxr_swapchain& operator=(const Openxr_swapchain&) = delete;
	Openxr_swapchain& operator=(Openxr_swapchain&& o) { swap(o); return *this; }

	uint32_t get_length(){
		uint32_t len = 0;
		//TODO error check
		xrEnumerateSwapchainImages(swapchain, 0, &len, nullptr);

		return len;
	}

	void swap(Openxr_swapchain& o) noexcept{
		std::swap(swapchain, o.swapchain);
		std::swap(session, o.session);
	}

	XrSwapchain get(){ return swapchain; }

private:
	XrSwapchain swapchain = XR_NULL_HANDLE;
	XrSession session = XR_NULL_HANDLE;
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

	XrSwapchain get() { return xr_swapchain.get(); }

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

class Openxr_local_space {
public:
	Openxr_local_space(XrSession session){
		XrPosef origin{};
	   	origin.orientation.x = 0.0;
	   	origin.orientation.y = 0.0;
	   	origin.orientation.z = 0.0;
	   	origin.orientation.w = 1.0;

		XrReferenceSpaceCreateInfo space_create_info;
		space_create_info.type = XR_TYPE_REFERENCE_SPACE_CREATE_INFO;
		space_create_info.next = NULL;
		space_create_info.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
		space_create_info.poseInReferenceSpace = origin;

		xrCreateReferenceSpace(session, &space_create_info, &space);
	}

	~Openxr_local_space(){
		xrDestroySpace(space);
	}

	Openxr_local_space(const Openxr_local_space&) = delete;
	Openxr_local_space(Openxr_local_space&&) = delete;
	Openxr_local_space& operator=(const Openxr_local_space&) = delete;
	Openxr_local_space& operator=(Openxr_local_space&&) = delete;

	XrSpace get() { return space; }

private:
	XrSpace space;
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

	Scene scene;

	std::chrono::steady_clock::time_point last_frame;

	std::mutex lock;
	std::condition_variable frame_consumed_cv;
	std::condition_variable new_frame_ready_cv;
	std::queue<video_frame *> free_frame_queue;
        std::queue<video_frame *> frame_queue;
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

static glm::mat4 get_proj_mat(const XrFovf& fov, float zNear, float zFar){
	glm::mat4 res{};

	float tanLeft = glm::tan(fov.angleLeft);
	float tanRight = glm::tan(fov.angleRight);
	float tanUp = glm::tan(fov.angleUp);
	float tanDown = glm::tan(fov.angleDown);

	res[0][0] = 2.f / (tanRight - tanLeft);
	res[1][1] = 2.f / (tanUp - tanDown);
	res[2][0] = (tanRight + tanLeft) / (tanRight - tanLeft); 
	res[2][1] = (tanUp + tanDown) / (tanUp - tanDown);
	res[2][2] = -(zFar + zNear) / (zFar - zNear);
	res[2][3] = -1;
	res[3][2] = -(2.f * zFar * zNear) / (zFar - zNear);

	return res;
}

static void display_xrgl_run(void *state){
        PROFILE_FUNC;

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

	Openxr_local_space space(session.get());

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

	size_t view_count = config_views.size();
	std::vector<XrCompositionLayerProjectionView> projection_views(view_count);

	XrCompositionLayerProjection projection_layer;
	projection_layer.type = XR_TYPE_COMPOSITION_LAYER_PROJECTION;
	projection_layer.next = nullptr;
	projection_layer.layerFlags;
	projection_layer.space = space.get();
	projection_layer.viewCount = view_count;
	projection_layer.views = projection_views.data();

	std::vector<XrView> views(view_count);

	for(auto& view : views){
		view.type = XR_TYPE_VIEW;
		view.next = nullptr;
	}

	bool running = true;
	while(running){
		std::unique_lock<std::mutex> lk(s->lock);
                video_frame *frame = nullptr;
                if(s->frame_queue.size() > 0){
                        PROFILE_DETAIL("put_frame");
                        frame = s->frame_queue.front();
                        s->frame_queue.pop();
                        s->frame_consumed_cv.notify_one();
                        if(!frame){
                                running = false;
                        }
                }
		lk.unlock();

		if(frame){
			s->scene.put_frame(frame);

			lk.lock();
			s->free_frame_queue.push(frame);
			lk.unlock();
		} 

		XrResult result;

		XrFrameState frame_state{};
		frame_state.type = XR_TYPE_FRAME_STATE;
		frame_state.next = nullptr;

		XrFrameWaitInfo frame_wait_info{};
		frame_wait_info.type = XR_TYPE_FRAME_WAIT_INFO;
		frame_wait_info.next = nullptr;

                PROFILE_DETAIL("wait frame");
		result = xrWaitFrame(session.get(), &frame_wait_info, &frame_state);
		if (!XR_SUCCEEDED(result)){
			log_msg(LOG_LEVEL_ERROR, "Failed to xrWaitFrame\n");
			break;
		}

		XrEventDataBuffer xr_event{};
		xr_event.type = XR_TYPE_EVENT_DATA_BUFFER;
		xr_event.next = nullptr;

		result = xrPollEvent(s->xr_state.instance.get(), &xr_event);
		//TODO process events

		XrViewLocateInfo view_locate_info;
	   	view_locate_info.type = XR_TYPE_VIEW_LOCATE_INFO;
		view_locate_info.displayTime = frame_state.predictedDisplayTime;
		view_locate_info.space = space.get();

		XrViewState view_state;
	   	view_state.type = XR_TYPE_VIEW_STATE;
	   	view_state.next = nullptr;

		uint32_t located_views = 0;
		result = xrLocateViews(session.get(),
				&view_locate_info,
				&view_state,
				view_count,
				&located_views,
				views.data());

		if (!XR_SUCCEEDED(result)){
			log_msg(LOG_LEVEL_ERROR, "Failed to locate views!\n");
			break;
		}

                /*
		printf("View: %f %f %f %f, %f %f %f, fov = %f %f %f %f\n",
				views[1].pose.orientation.x,
				views[1].pose.orientation.y,
				views[1].pose.orientation.z,
				views[1].pose.orientation.w,
				views[1].pose.position.x,
				views[1].pose.position.y,
				views[1].pose.position.z,
				views[1].fov.angleLeft,
				views[1].fov.angleRight,
				views[1].fov.angleUp,
				views[1].fov.angleDown);
                                */

		XrFrameBeginInfo frame_begin_info;
		frame_begin_info.type = XR_TYPE_FRAME_BEGIN_INFO;
		frame_begin_info.next = nullptr;

                PROFILE_DETAIL("begin frame");
		result = xrBeginFrame(session.get(), &frame_begin_info);
		if (!XR_SUCCEEDED(result)){
			log_msg(LOG_LEVEL_ERROR, "Failed to begin frame!\n");
			break;
		}

		for(unsigned i = 0; i < view_count; i++){
                        PROFILE_DETAIL("render view");

			XrSwapchainImageAcquireInfo swapchain_image_acquire_info;
			swapchain_image_acquire_info.type = XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO;
			swapchain_image_acquire_info.next = nullptr;

			uint32_t buf_idx;
			result = xrAcquireSwapchainImage(
					swapchains[i].get(),
					&swapchain_image_acquire_info,
					&buf_idx);

			if(!XR_SUCCEEDED(result)){
				log_msg(LOG_LEVEL_ERROR, "Failed to acquire swapchain image!\n");
				break;
			}

			XrSwapchainImageWaitInfo swapchain_image_wait_info;
			swapchain_image_wait_info.type = XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO;
			swapchain_image_wait_info.next = nullptr;
			swapchain_image_wait_info.timeout = 1000;

			result = xrWaitSwapchainImage(swapchains[i].get(), &swapchain_image_wait_info);
			if(!XR_SUCCEEDED(result)){
				log_msg(LOG_LEVEL_ERROR, "failed to wait for swapchain image!\n");
				break;
			}

			projection_views[i].type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW;
			projection_views[i].next = nullptr;
			projection_views[i].pose = views[i].pose;
			projection_views[i].fov = views[i].fov;
			projection_views[i].subImage.swapchain = swapchains[i].get();
			projection_views[i].subImage.imageArrayIndex = buf_idx;
			projection_views[i].subImage.imageRect.offset.x = 0;
			projection_views[i].subImage.imageRect.offset.y = 0;
			projection_views[i].subImage.imageRect.extent.width =
				config_views[i].recommendedImageRectWidth;
			projection_views[i].subImage.imageRect.extent.height =
				config_views[i].recommendedImageRectHeight;

	
			unsigned w = config_views[i].recommendedImageRectWidth;
			unsigned h = config_views[i].recommendedImageRectHeight;

			glm::mat4 projMat = get_proj_mat(views[i].fov, 0.05f, 100.f);
			//glm::mat4 projMat = glm::perspective(glm::radians(70.f), (float) w /h, 0.1f, 300.f);
			const auto& rot = views[i].pose.orientation;
			glm::mat4 viewMat = glm::mat4_cast(glm::quat(rot.w, rot.x, rot.y, rot.z));
			glm::mat4 pvMat = projMat * glm::inverse(viewMat);

			//TODO Render frame

			auto framebuffer = swapchains[i].get_framebuffer(buf_idx).get();
			glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
			if(i % 2){
				glClearColor(0.f, 0.f, 1.f, 1.f);
			} else {
				glClearColor(0.f, 1.f, 0.f, 1.0f);
			}
			glClear(GL_COLOR_BUFFER_BIT);

			s->scene.render(w, h, pvMat);

			glBindFramebuffer(GL_FRAMEBUFFER, 0);

			if(i % 2){
				glBlitNamedFramebuffer((GLuint)framebuffer, // readFramebuffer
						(GLuint)0,    // backbuffer     // drawFramebuffer
						(GLint)0,     // srcX0
						(GLint)0,     // srcY0
						(GLint)w,     // srcX1
						(GLint)h,     // srcY1
						(GLint)0,     // dstX0
						(GLint)0,     // dstY0
						(GLint)w / 2, // dstX1
						(GLint)h / 2, // dstY1
						(GLbitfield)GL_COLOR_BUFFER_BIT, // mask
						(GLenum)GL_LINEAR);              // filter

				SDL_GL_SwapWindow(s->window.sdl_window);
			}

                        PROFILE_DETAIL("glFinish");
			glFinish();

                        PROFILE_DETAIL("release swapchain");
			XrSwapchainImageReleaseInfo swapchain_image_release_info;
			swapchain_image_release_info.type = XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO;
			swapchain_image_release_info.next = nullptr;

			result = xrReleaseSwapchainImage(
					swapchains[i].get(),
					&swapchain_image_release_info);

			if (!XR_SUCCEEDED(result)){
				log_msg(LOG_LEVEL_ERROR, "Failed to release swapchain image!\n");
				break;
			}
		}

		const XrCompositionLayerBaseHeader *composition_layers = (const XrCompositionLayerBaseHeader *) &projection_layer; 
		XrFrameEndInfo frame_end_info;
		frame_end_info.type = XR_TYPE_FRAME_END_INFO;
		frame_end_info.displayTime = frame_state.predictedDisplayTime;
		frame_end_info.layerCount = 1;
		frame_end_info.layers = &composition_layers;
		frame_end_info.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
		frame_end_info.next = nullptr;

                PROFILE_DETAIL("End Frame");
		result = xrEndFrame(session.get(), &frame_end_info);
		if (!XR_SUCCEEDED(result)){
			log_msg(LOG_LEVEL_ERROR, "Failed to end frame!\n");
			break;
		}

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

        struct video_frame *buffer = vf_alloc_desc_data(s->current_desc);
        clear_video_buffer(reinterpret_cast<unsigned char *>(buffer->tiles[0].data),
                        vc_get_linesize(buffer->tiles[0].width, buffer->color_spec),
                        vc_get_linesize(buffer->tiles[0].width, buffer->color_spec),
                        buffer->tiles[0].height,
                        buffer->color_spec);
        return buffer;
}

static int display_xrgl_putf(void *state, struct video_frame *frame, int nonblock) {
        struct state_xrgl *s = static_cast<state_xrgl *>(state);

        std::unique_lock<std::mutex> lk(s->lock);

        if(!frame) {
                s->frame_queue.push(frame);
                lk.unlock();
                s->new_frame_ready_cv.notify_one();
                return 0;
        }

        if (nonblock == PUTF_DISCARD) {
                vf_recycle(frame);
                return 0;
        }
        if (s->frame_queue.size() >= MAX_BUFFER_SIZE && nonblock == PUTF_NONBLOCK) {
                vf_recycle(frame);
                s->free_frame_queue.push(frame);
                return 1;
        }
        s->frame_consumed_cv.wait(lk, [s]{return s->frame_queue.size() < MAX_BUFFER_SIZE;});
        s->frame_queue.push(frame);

        lk.unlock();
        s->new_frame_ready_cv.notify_one();

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
		UYVY,
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
