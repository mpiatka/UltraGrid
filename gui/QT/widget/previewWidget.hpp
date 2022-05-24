#ifndef PREVIEWWIDGET_HPP
#define PREVIEWWIDGET_HPP

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QSharedMemory>
#include <QOpenGLVertexArrayObject>
#include <QTimer>

#include "ipc_frame.h"
#include "ipc_frame_unix.h"

class PreviewWidget : public QOpenGLWidget{
public:
	PreviewWidget(QWidget *parent) :
		QOpenGLWidget(parent),
		ipc_frame(ipc_frame_new())
   	{
		connect(&timer, SIGNAL(timeout()), this, SLOT(update()));
  	}

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

	Ipc_frame_uniq ipc_frame;
	Ipc_frame_reader_uniq ipc_frame_reader;
	QTimer timer;
};

#endif
