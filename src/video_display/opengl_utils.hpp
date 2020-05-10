#ifndef OPENGL_UTILS_HPP
#define OPENGL_UTILS_HPP

#include <X11/Xlib.h>
#include <GL/glew.h>
#include <GL/glx.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

#include "types.h"

class GlProgram{
public:
	GlProgram(const char *vert_src, const char *frag_src);
	~GlProgram();

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
	~Model();

	GLuint get_vao() const { return vao; }

	void render();

	static Model get_sphere();
	static Model get_quad();

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
	Texture();
	~Texture();

	GLuint get() const { return tex_id; }

	void allocate(int w, int h, GLenum fmt);

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

	void attach_texture(GLuint tex);

	void attach_texture(const Texture& tex){
		attach_texture(tex.get());
	}


private:
	void swap(Framebuffer& o){
		std::swap(fbo, o.fbo);
	}

	GLuint fbo = 0;
};

class Yuv_convertor{
public:
    Yuv_convertor();

	void put_frame(const video_frame *f);

	void attach_texture(const Texture& tex){
		fbuf.attach_texture(tex);
	}

private:
	GlProgram program;// = GlProgram(vert_src, yuv_conv_frag_src);
	Model quad;// = Model::get_quad();
	Framebuffer fbuf;
	Texture yuv_tex;
};

struct Scene{
    Scene();

	void render(int width, int height);
	void render(int width, int height, const glm::mat4& pvMat);

	void put_frame(const video_frame *f);

	void rotate(float dx, float dy);

	GlProgram program;// = GlProgram(persp_vert_src, persp_frag_src);
	Model model;// = Model::get_sphere();
	Texture texture;
	Framebuffer framebuffer;
	Yuv_convertor conv;
	float rot_x = 0;
	float rot_y = 0;
	float fov = 55;

	int width, height;
};

#endif
