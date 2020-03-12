
#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <chrono>

#include <assert.h>
#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <SDL2/SDL_opengl.h>

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

static const GLfloat rectangle[] = {
	 1.0f,  1.0f,  1.0f,  1.0f,
	-1.0f,  1.0f,  0.0f,  1.0f,
	-1.0f, -1.0f,  0.0f,  0.0f,

	 1.0f,  1.0f,  1.0f,  1.0f,
	-1.0f, -1.0f,  0.0f,  0.0f,
	 1.0f, -1.0f,  1.0f,  0.0f
};

static unsigned char pixels[] = {
	255, 0, 0,   0, 255, 0,   0, 0, 255,   255, 255, 255,
	255, 0, 0,   0, 255, 0,   0, 0, 255,   255, 255, 255,
	255, 0, 0,   0, 255, 0,   0, 0, 255,   255, 255, 255,
	255, 0, 0,   0, 255, 0,   0, 0, 255,   255, 255, 255
};

static const char *vert_src = R"END(
#version 330 core
layout(location = 0) in vec2 vert_pos;
layout(location = 1) in vec2 vert_uv;

out vec2 UV;

uniform vec2 scale_vec;

void main(){
	gl_Position = vec4(vert_pos, 0.0f, 1.0f);
	UV = vert_uv;
}
)END";

static const char *frag_src = R"END(
#version 330 core
in vec2 UV;
out vec3 color;
uniform sampler2D tex;
void main(){
	color = texture(tex, UV).rgb;
}
)END";

static const char *persp_vert_src = R"END(
#version 330 core
layout(location = 0) in vec3 vert_pos;
layout(location = 1) in vec2 vert_uv;

out vec2 UV;

uniform mat4 pv_mat;

void main(){
	gl_Position = pv_mat * vec4(vert_pos, 1.0f);
	UV = vert_uv;
}
)END";

static const char *persp_frag_src = R"END(
#version 330 core
in vec2 UV;
out vec3 color;
uniform sampler2D tex;
void main(){
	color = texture(tex, UV).rgb;
}
)END";

static const char *yuv_conv_frag_src = R"END(
#version 330 core
layout(location = 0) out vec4 color;
in vec2 UV;
uniform sampler2D tex;
uniform float width;

void main(){
	vec4 yuv;
	yuv.rgba  = texture2D(tex, UV).grba;
	if(UV.x * width / 2.0 - floor(UV.x * width / 2.0) > 0.5)
		yuv.r = yuv.a;

	yuv.r = 1.1643 * (yuv.r - 0.0625);
	yuv.g = yuv.g - 0.5;
	yuv.b = yuv.b - 0.5;
	float tmp; // this is a workaround over broken Gallium3D with Nouveau in U14.04 (and perhaps others)
	tmp = -0.2664 * yuv.b;
	tmp = 2.0 * tmp;
	color.r = yuv.r + 1.7926 * yuv.b;
	color.g = yuv.r - 0.2132 * yuv.g - 0.5328 * yuv.b;
	color.b = yuv.r + 2.1124 * yuv.g;
	color.a = 1.0;
}
)END";

static void compileShader(GLuint shaderId){
	glCompileShader(shaderId);

	GLint ret = GL_FALSE;
	int len;

	glGetShaderiv(shaderId, GL_COMPILE_STATUS, &ret);
	glGetShaderiv(shaderId, GL_INFO_LOG_LENGTH, &len);
	if (len > 0){
		std::vector<char> errorMsg(len+1);
		glGetShaderInfoLog(shaderId, len, NULL, &errorMsg[0]);
		printf("%s\n", errorMsg.data());
	}
}


struct Window{
	Window() : Window("UltraGrid VR",
			SDL_WINDOWPOS_CENTERED,
			SDL_WINDOWPOS_CENTERED,
			640,
			480,
			SDL_WINDOW_OPENGL) {  }

	Window(const char *title, int x, int y, int w, int h, SDL_WindowFlags flags) :
	width(w), height(h)
	{
		SDL_InitSubSystem(SDL_INIT_VIDEO);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);

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


	~Window(){
		SDL_DestroyWindow(sdl_window);
		SDL_QuitSubSystem(SDL_INIT_VIDEO);
	}


	SDL_Window *sdl_window;
	SDL_GLContext sdl_gl_context;
	int width;
	int height;
};

static std::vector<float> gen_sphere_vertices(int r, int latitude_n, int longtitude_n);
static std::vector<unsigned> gen_sphere_indices(int latitude_n, int longtitude_n);

class GlProgram{
public:
	GlProgram(const char *vert_src, const char *frag_src){
		GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
		GLuint fragShader = glCreateShader(GL_FRAGMENT_SHADER);

		glShaderSource(vertexShader, 1, &vert_src, NULL);
		compileShader(vertexShader);
		glShaderSource(fragShader, 1, &frag_src, NULL);
		compileShader(fragShader);

		program = glCreateProgram();
		glAttachShader(program, vertexShader);
		glAttachShader(program, fragShader);
		glLinkProgram(program);
		//glUseProgram(program);

		glDetachShader(program, vertexShader);
		glDetachShader(program, fragShader);
		glDeleteShader(vertexShader);
		glDeleteShader(fragShader);
	}

	~GlProgram() {
		glUseProgram(0);
		glDeleteProgram(program);
	}

	GLuint get() { return program; }

	GlProgram(const GlProgram&) = delete;
	GlProgram(GlProgram&& o) { swap(o); }
	GlProgram& operator=(const GlProgram&) = delete;
	GlProgram& operator=(GlProgram&& o) { swap(o); return *this; }

private:
	void swap(GlProgram& o){
		std::swap(program, o.program);
	}
	GLuint program = 0;
};

class Model{
public:
	Model(const Model&) = delete;
	Model(Model&& o) { swap(o); }
	Model& operator=(const Model&) = delete;
	Model& operator=(Model&& o) { swap(o); return *this; }
	~Model(){
		glDeleteBuffers(1, &vbo);
		glDeleteBuffers(1, &elem_buf);
		glDeleteVertexArrays(1, &vao);
	}

	GLuint get_vao() const { return vao; }

	void render(){
		Profile_timer t(Profiler_thread_inst::get_instance(), "Model render");
		glBindVertexArray(vao);
		if(elem_buf != 0){
			glDrawElements(GL_TRIANGLES, indices_num, GL_UNSIGNED_INT, (void *) 0);
		} else {
			glDrawArrays(GL_TRIANGLES, 0, indices_num);
		}

		glBindVertexArray(0);
	}

	static Model get_sphere(){
		Model model;
		glGenVertexArrays(1, &model.vao);
		glBindVertexArray(model.vao);

		auto vertices = gen_sphere_vertices(1, 64, 64);
		auto indices = gen_sphere_indices(64, 64);

		glGenBuffers(1, &model.vbo);
		glBindBuffer(GL_ARRAY_BUFFER, model.vbo);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

		glGenBuffers(1, &model.elem_buf);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, model.elem_buf);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned), indices.data(), GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, model.vbo);
		glVertexAttribPointer(
				0,
				3,
				GL_FLOAT,
				GL_FALSE,
				5 * sizeof(float),
				(void*)0
				);
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, model.vbo);
		glVertexAttribPointer(
				1,
				2,
				GL_FLOAT,
				GL_FALSE,
				5 * sizeof(float),
				(void*)(3 * sizeof(float))
				);

		glBindVertexArray(0);
		model.indices_num = indices.size();

		return model;
	}

	static Model get_quad(){
		Model model;
		glGenVertexArrays(1, &model.vao);
		glBindVertexArray(model.vao);

		glGenBuffers(1, &model.vbo);
		glBindBuffer(GL_ARRAY_BUFFER, model.vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(rectangle), rectangle, GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, model.vbo);
		glVertexAttribPointer(
				0,
				2,
				GL_FLOAT,
				GL_FALSE,
				4 * sizeof(float),
				(void*)0
				);
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, model.vbo);
		glVertexAttribPointer(
				1,
				2,
				GL_FLOAT,
				GL_FALSE,
				4 * sizeof(float),
				(void*)(2 * sizeof(float))
				);

		glBindVertexArray(0);
		model.indices_num = 6;

		return model;
	}

private:
	Model() = default;
	void swap(Model& o){
		std::swap(vao, o.vao);
		std::swap(vbo, o.vbo);
		std::swap(elem_buf, o.elem_buf);
		std::swap(indices_num, o.indices_num);
	}
	GLuint vao = 0;
	GLuint vbo = 0;
	GLuint elem_buf = 0;
	GLsizei indices_num = 0;
};

class Texture{
public:
	Texture(){
		glGenTextures(1, &tex_id);
		glBindTexture(GL_TEXTURE_2D, tex_id);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 4, 4, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}

	~Texture(){
		glDeleteTextures(1, &tex_id);
	}

	GLuint get() const { return tex_id; }

	void allocate(int w, int h, GLenum fmt) {
		if(w != width || h != height || fmt != format){
			Profile_timer t(Profiler_thread_inst::get_instance(), "Tex allocate");
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

	void attach_texture(const Texture& tex){
		glBindTexture(GL_TEXTURE_2D, tex.get());
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glBindTexture(GL_TEXTURE_2D, 0);

		glBindFramebuffer(GL_FRAMEBUFFER, fbo);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex.get(), 0);
		assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

private:
	void swap(Framebuffer& o){
		std::swap(fbo, o.fbo);
	}

	GLuint fbo = 0;
};

class Yuv_convertor{
public:
	void put_frame(const video_frame *f){
		Profile_timer t(Profiler_thread_inst::get_instance(), "Yuv put_frame");
		glUseProgram(program.get());
		glBindFramebuffer(GL_FRAMEBUFFER, fbuf.get());
		glViewport(0, 0, f->tiles[0].width, f->tiles[0].height);
		glClear(GL_COLOR_BUFFER_BIT);

		glBindTexture(GL_TEXTURE_2D, yuv_tex.get());
		yuv_tex.allocate(f->tiles[0].width / 2, f->tiles[0].height, GL_RGBA);

		{
		Profile_timer t2(Profiler_thread_inst::get_instance(), "TexSubImage");
		glTexSubImage2D(GL_TEXTURE_2D, 0,
				0, 0, f->tiles[0].width / 2, f->tiles[0].height,
						GL_RGBA, GL_UNSIGNED_BYTE, f->tiles[0].data);
		}

		GLuint w_loc = glGetUniformLocation(program.get(), "width");
		glUniform1f(w_loc, f->tiles[0].width);

		quad.render();

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glUseProgram(0);
	}

	void attach_texture(const Texture& tex){
		fbuf.attach_texture(tex);
	}

private:
	GlProgram program = GlProgram(vert_src, yuv_conv_frag_src);
	Model quad = Model::get_quad();
	Framebuffer fbuf;
	Texture yuv_tex;
};

struct Scene{
	void render(int width, int height){
		Profile_timer t(Profiler_thread_inst::get_instance(), "Scene render");
		glUseProgram(program.get());
		float aspect_ratio = static_cast<float>(width) / height;
		glViewport(0, 0, width, height);
		glm::mat4 projMat = glm::perspective(glm::radians(fov),
				aspect_ratio,
				0.1f,
				300.0f);
		glm::mat4 viewMat(1.f);
		viewMat = glm::rotate(viewMat, glm::radians(rot_y), {1.f, 0, 0});
		viewMat = glm::rotate(viewMat, glm::radians(rot_x), {0.f, 1, 0});
		glm::mat4 pvMat = projMat * viewMat;

		GLuint pvLoc;
		pvLoc = glGetUniformLocation(program.get(), "pv_mat");
		glUniformMatrix4fv(pvLoc, 1, GL_FALSE, glm::value_ptr(pvMat));

		glBindTexture(GL_TEXTURE_2D, texture.get());
		model.render();
	}

	void put_frame(const video_frame *f){
		Profile_timer t(Profiler_thread_inst::get_instance(), "Scene put_frame");

		glBindTexture(GL_TEXTURE_2D, texture.get());
		texture.allocate(f->tiles[0].width, f->tiles[0].height, GL_RGB);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		switch(f->color_spec){
			case UYVY:
				conv.attach_texture(texture);
				conv.put_frame(f);
				break;
			case RGB:
				glTexSubImage2D(GL_TEXTURE_2D, 0,
						0, 0, f->tiles[0].width, f->tiles[0].height,
						GL_RGB, GL_UNSIGNED_BYTE, f->tiles[0].data);
				break;
			case RGBA:
				glTexSubImage2D(GL_TEXTURE_2D, 0,
						0, 0, f->tiles[0].width, f->tiles[0].height,
						GL_RGBA, GL_UNSIGNED_BYTE, f->tiles[0].data);
			default:
				break;
		}
	}

	void rotate(float dx, float dy){
		rot_x -= dx;
		rot_y -= dy;

		if(rot_y > 90) rot_y = 90;
		if(rot_y < -90) rot_y = -90;
	}

	GlProgram program = GlProgram(persp_vert_src, persp_frag_src);
	Model model = Model::get_sphere();
	Texture texture;
	Framebuffer framebuffer;
	Yuv_convertor conv;
	float rot_x = 0;
	float rot_y = 0;
	float fov = 55;

	int width, height;
};

struct state_vr{
	Window window;

	bool running = false;

	unsigned sdl_frame_event;
	unsigned sdl_redraw_event;

	video_desc current_desc;
	int buffered_frames_count;

	Scene scene;

	int max_fps = 60;

	SDL_TimerID redraw_timer = 0;
	bool redraw_needed = false;

	std::chrono::steady_clock::time_point last_frame;

	std::mutex lock;
	std::condition_variable frame_consumed_cv;
	std::queue<video_frame *> free_frame_queue;
};

static std::vector<float> gen_sphere_vertices(int r, int latitude_n, int longtitude_n){
	std::vector<float> verts;

	float lat_step = PI_F / latitude_n;
	float long_step = 2 * PI_F / longtitude_n;

	for(int i = 0; i < latitude_n + 1; i++){
		float y = std::cos(i * lat_step) * r;
		float y_slice_r = std::sin(i * lat_step) * r;

		//The first and last vertex on the y slice circle are in the same place
		for(int j = 0; j < longtitude_n + 1; j++){
			float x = std::sin(j * long_step) * y_slice_r;
			float z = std::cos(j * long_step) * y_slice_r;
			verts.push_back(x);
			verts.push_back(y);
			verts.push_back(z);

			float u = 1 - static_cast<float>(j) / longtitude_n;
			float v = static_cast<float>(i) / latitude_n;
			verts.push_back(u);
			verts.push_back(v);
		}
	}

	return verts;
}

//Generate indices for sphere
//Faces facing inwards have counter-clockwise vertex order
static std::vector<unsigned> gen_sphere_indices(int latitude_n, int longtitude_n){
	std::vector<unsigned int> indices;

	for(int i = 0; i < latitude_n; i++){
		int slice_idx = i * (latitude_n + 1);
		int next_slice_idx = (i + 1) * (latitude_n + 1);

		for(int j = 0; j < longtitude_n; j++){
			//Since the top and bottom slices are circles with radius 0,
			//we only need one triangle for those
			if(i != latitude_n - 1){
				indices.push_back(slice_idx + j + 1);
				indices.push_back(next_slice_idx + j);
				indices.push_back(next_slice_idx + j + 1);
			}

			if(i != 0){
				indices.push_back(slice_idx + j + 1);
				indices.push_back(slice_idx + j);
				indices.push_back(next_slice_idx + j);
			}
		}
	}

	return indices;
}

static void * display_vr_init(struct module *parent, const char *fmt, unsigned int flags) {
	state_vr *s = new state_vr();
	s->sdl_frame_event = SDL_RegisterEvents(1);
	s->sdl_redraw_event = SDL_RegisterEvents(1);

	return s;
}

static void draw(state_vr *s){
	auto now = std::chrono::steady_clock::now();
	if(std::chrono::duration_cast<std::chrono::milliseconds>(now - s->last_frame).count() < 16)
		return;

	s->last_frame = now;

	glClear(GL_COLOR_BUFFER_BIT);
	
	s->scene.render(s->window.width, s->window.height);

	SDL_GL_SwapWindow(s->window.sdl_window);
}

static Uint32 redraw_callback(Uint32 interval, void *param){
	int event_id = *static_cast<int *>(param);
	SDL_Event event;
	event.type = event_id;
	SDL_PushEvent(&event);

	return interval;
}

static void redraw(state_vr *s){
	if(!s->redraw_timer){
		s->redraw_timer = SDL_AddTimer(1000 / s->max_fps, redraw_callback, &s->sdl_redraw_event);
		draw(s);
	} else {
		s->redraw_needed = true;
	}
}

static void handle_window_event(state_vr *s, SDL_Event *event){
	if(event->window.event == SDL_WINDOWEVENT_RESIZED){
		glViewport(0, 0, event->window.data1, event->window.data2);
		s->window.width = event->window.data1;
		s->window.height = event->window.data2;
		redraw(s);
	}
}

static void handle_user_event(state_vr *s, SDL_Event *event){
	if(event->type == s->sdl_frame_event){
		std::unique_lock<std::mutex> lk(s->lock);
		s->buffered_frames_count -= 1;
		lk.unlock();
		s->frame_consumed_cv.notify_one();

		video_frame *frame = static_cast<video_frame *>(event->user.data1);

		if(frame){
			s->scene.put_frame(frame);

			lk.lock();
			s->free_frame_queue.push(frame);
			lk.unlock();
		} else {
			//poison
			s->running = false;
		}
		redraw(s);
	} else if(event->type == s->sdl_redraw_event){
		if(s->redraw_needed){
			draw(s);
			s->redraw_needed = false;
		} else {
			SDL_RemoveTimer(s->redraw_timer);
			s->redraw_timer = 0;
		}
	}
}

static void display_vr_run(void *state) {
	state_vr *s = static_cast<state_vr *>(state);

	draw(s);

	s->running = true;
	while(s->running){
		SDL_Event event;
		if (!SDL_WaitEvent(&event)) {
			continue;
		}

		switch(event.type){
			case SDL_MOUSEMOTION:
				if(event.motion.state & SDL_BUTTON_LMASK){
					s->scene.rotate(event.motion.xrel / 8.f,
							event.motion.yrel / 8.f);

					redraw(s);
				}
				break;
			case SDL_MOUSEWHEEL:
				s->scene.fov -= event.wheel.y;
				redraw(s);
				break;
			case SDL_WINDOWEVENT:
				handle_window_event(s, &event);
				break;
			case SDL_QUIT:
				exit_uv(0);
				break;
			default: 
				if(event.type >= SDL_USEREVENT)
					handle_user_event(s, &event);
				break;
		}
	}
}

static void display_vr_done(void *state) {
	state_vr *s = static_cast<state_vr *>(state);

	delete s;
}

static struct video_frame * display_vr_getf(void *state) {
	struct state_vr *s = static_cast<state_vr *>(state);

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

static int display_vr_putf(void *state, struct video_frame *frame, int nonblock) {
	struct state_vr *s = static_cast<state_vr *>(state);

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
	SDL_Event event;
	event.type = s->sdl_frame_event;
	event.user.data1 = frame;
	SDL_PushEvent(&event);

	return 0;
}

static int display_vr_reconfigure(void *state, struct video_desc desc) {
	state_vr *s = static_cast<state_vr *>(state);

	s->current_desc = desc;
	return 1;
}

static int display_vr_get_property(void *state, int property, void *val, size_t *len) {
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
        NULL,
        NULL,
};

REGISTER_MODULE(vr_disp, &display_vr_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);
