#ifndef PREVIEWWIDGET_HPP
#define PREVIEWWIDGET_HPP

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QSharedMemory>
#include <QOpenGLVertexArrayObject>
#include <QTimer>

#include "shared_mem_frame.hpp"

class PreviewWidget : public QOpenGLWidget{
public:
	PreviewWidget(QWidget *parent);
	~PreviewWidget();

	void setKey(const char *key);
	void start();
	void stop();

protected:
	void initializeGL();
	void resizeGL(int w, int h);
	void paintGL();

private:
	bool loadFrame();

	GLuint vertexBuffer = 0;
	GLuint program = 0;
	GLuint texture = 0;

	QOpenGLVertexArrayObject vao;

	GLfloat scaleVec[2];
	int vidW = 0;
	int vidH = 0;
	int width = 0;
	int height = 0;

	void setVidSize(int w, int h);
	void calculateScale();

	Shared_mem shared_mem;
	QTimer timer;
};

#endif
