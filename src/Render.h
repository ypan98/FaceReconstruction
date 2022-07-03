//#pragma once
#define GLEW_STATIC

#include "Eigen.h"
#include "FaceModel.h"
#include "opencv2/core/core.hpp"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <chrono>

cv::Mat render(FaceModel face_model, Matrix4f projection_matrix, int viewport_width, int viewport_height, bool enable_backface_culling = true, bool enable_near_clipping = true, bool enable_far_clipping = true)
{	
	/* Initialize the library */

	GLFWwindow* window;

	glfwInit();

	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
	window = glfwCreateWindow(viewport_width, viewport_height, "face", NULL, NULL);

	glfwMakeContextCurrent(window);

	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glShadeModel(GL_SMOOTH);

	glewInit();

	MatrixXf model_vertices(4, face_model.getNumVertices());
	model_vertices.block(0, 0, 3, face_model.getNumVertices()) = face_model.getShapeMean().reshaped(3, face_model.getNumVertices());
	model_vertices.row(3) = VectorXf::Ones(face_model.getNumVertices());
	MatrixXf clipspace_vertices_coord = (projection_matrix * model_vertices).transpose();

	GLuint color;
	glGenTextures(1, &color);
	glBindTexture(GL_TEXTURE_2D, color);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, viewport_width, viewport_height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);

	GLuint depth;
	glGenRenderbuffers(1, &depth);
	glBindRenderbuffer(GL_RENDERBUFFER, depth);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, viewport_width, viewport_height);

	GLuint fbo;
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);

	GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
	glDrawBuffers(1, DrawBuffers);
	
	glViewport(0, 0, viewport_width, viewport_height);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glShadeModel(GL_SMOOTH);

	for (int i = 0; i < face_model.getTriangulation().rows(); ++i) {
		for (int v = 0; v < 3; v++) {
			int vertex_index = face_model.getTriangulation()(i, v);
			float w = clipspace_vertices_coord(vertex_index, 3);
			glBegin(GL_TRIANGLES);
			glColor3d(face_model.getColorMean()[vertex_index * 3], face_model.getColorMean()[vertex_index * 3 + 1], face_model.getColorMean()[vertex_index * 3 + 2]);
			glVertex3d(clipspace_vertices_coord(vertex_index, 0) / w, clipspace_vertices_coord(vertex_index, 1) / w, clipspace_vertices_coord(vertex_index, 2) / w);
			glEnd();
		}
	}

	unsigned char* gl_texture_bytes = (unsigned char*)malloc(sizeof(unsigned char) * viewport_height * viewport_width * 3);
	glReadPixels(0, 0, viewport_height, viewport_width, 0x80E0, GL_UNSIGNED_BYTE, gl_texture_bytes);
	cv::Mat img(viewport_height, viewport_width, CV_8UC3, gl_texture_bytes);
	
	glDeleteFramebuffers(1, &fbo);
	glfwSwapBuffers(window);
	glfwPollEvents();
	return img;
};
