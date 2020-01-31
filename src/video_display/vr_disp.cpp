
#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <assert.h>
#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <SDL2/SDL_opengl.h>

#include "debug.h"
#include "host.h"
#include "keyboard_control.h" // K_*
#include "lib_common.h"
#include "messaging.h"
#include "module.h"
#include "video.h"
#include "video_display.h"
#include "video_display/splashscreen.h"

#define MAX_BUFFER_SIZE   1

static const GLfloat rectangle[] = {
	 1.0f,  1.0f,  1.0f,  0.0f,
	-1.0f,  1.0f,  0.0f,  0.0f,
	-1.0f, -1.0f,  0.0f,  1.0f,

	 1.0f,  1.0f,  1.0f,  0.0f,
	-1.0f, -1.0f,  0.0f,  1.0f,
	 1.0f, -1.0f,  1.0f,  1.0f
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

	Window(const char *title, int x, int y, int w, int h, SDL_WindowFlags flags){
		SDL_InitSubSystem(SDL_INIT_VIDEO);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);

		//TODO: Error handling
		sdl_window = SDL_CreateWindow("UltraGrid VR",
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
};

struct state_vr{
	Window window;

	int sdl_frame_event;

	video_desc current_desc;
	int buffered_frames_count;

	GLuint vao = 0;
	GLuint vbo = 0;
	GLuint program = 0;
	GLuint gl_texture = 0;

	std::mutex lock;
	std::condition_variable frame_consumed_cv;
	std::queue<video_frame *> free_frame_queue;
};

static void * display_vr_init(struct module *parent, const char *fmt, unsigned int flags) {
	state_vr *s = new state_vr();
	s->sdl_frame_event = SDL_RegisterEvents(1);

	return s;
}

static void draw(state_vr *s){
	glUseProgram(s->program);
	glBindVertexArray(s->vao);
	glClear(GL_COLOR_BUFFER_BIT);

	glBindTexture(GL_TEXTURE_2D, s->gl_texture);
	glDrawArrays(GL_TRIANGLES, 0, 6);

	glBindVertexArray(0);

	SDL_GL_SwapWindow(s->window.sdl_window);
}

static void handle_window_event(state_vr *s, SDL_Event *event){
	if(event->window.event == SDL_WINDOWEVENT_RESIZED){
		glViewport(0, 0, event->window.data1, event->window.data2);
		draw(s);
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
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame->tiles[0].width, frame->tiles[0].height, 0, GL_RGB, GL_UNSIGNED_BYTE, frame->tiles[0].data);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			lk.lock();
			s->free_frame_queue.push(frame);
			lk.unlock();
		}
		draw(s);

	}
}

static void initialize_scene(state_vr *s){
	glGenVertexArrays(1, &s->vao);
	glBindVertexArray(s->vao);


	glGenBuffers(1, &s->vbo);
	glBindBuffer(GL_ARRAY_BUFFER, s->vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(rectangle), rectangle, GL_STATIC_DRAW);

	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	GLuint fragShader = glCreateShader(GL_FRAGMENT_SHADER);

	glShaderSource(vertexShader, 1, &vert_src, NULL);
	compileShader(vertexShader);
	glShaderSource(fragShader, 1, &frag_src, NULL);
	compileShader(fragShader);

	s->program = glCreateProgram();
	glAttachShader(s->program, vertexShader);
	glAttachShader(s->program, fragShader);
	glLinkProgram(s->program);
	glUseProgram(s->program);

	glDetachShader(s->program, vertexShader);
	glDetachShader(s->program, fragShader);
	glDeleteShader(vertexShader);
	glDeleteShader(fragShader);

	glGenTextures(1, &s->gl_texture);
	glBindTexture(GL_TEXTURE_2D, s->gl_texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 4, 4, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, s->vbo);
	glVertexAttribPointer(
			0,
			2,
			GL_FLOAT,
			GL_FALSE,
			4 * sizeof(float),
			(void*)0
			);
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, s->vbo);
	glVertexAttribPointer(
			1,
			2,
			GL_FLOAT,
			GL_FALSE,
			4 * sizeof(float),
			(void*)(2 * sizeof(float))
			);

	glBindVertexArray(0);
}

static void display_vr_run(void *state) {
	state_vr *s = static_cast<state_vr *>(state);

	initialize_scene(s);
	draw(s);

	bool running = true;
	while(running){
		SDL_Event event;
		if (!SDL_WaitEvent(&event)) {
			continue;
		}

		switch(event.type){
			case SDL_WINDOWEVENT:
				handle_window_event(s, &event);
				break;
			case SDL_QUIT:
				running = false;
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
