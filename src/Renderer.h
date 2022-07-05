//#pragma once
#define GLEW_STATIC

#include "Eigen.h"
#include "FaceModel.h"
#include "opencv2/core/core.hpp"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <chrono>

class Renderer{
public:
	static Renderer& Get() {
		return s_instance;
	}

	cv::Mat render(Face face, Matrix4f projection_matrix, int viewport_width, int viewport_height, bool enable_backface_culling = true, bool enable_near_clipping = true, bool enable_far_clipping = true)
	{
		MatrixXf model_vertices(4, face.getFaceModel().getNumVertices());
		model_vertices.block(0, 0, 3, face.getFaceModel().getNumVertices()) = face.calculateVerticesDefault().reshaped(3, face.getFaceModel().getNumVertices());
		model_vertices.row(3) = VectorXf::Ones(face.getFaceModel().getNumVertices());
		MatrixXf clipspace_vertices_coord = (projection_matrix * model_vertices).block(0, 0, 3, face.getFaceModel().getNumVertices());
		Eigen::Matrix<unsigned int, -1, -1> triangulation = face.getFaceModel().getTriangulation().transpose().cast <unsigned int>();

		GLuint color;
		glGenTextures(1, &color);
		glBindTexture(GL_TEXTURE_2D, color);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, viewport_width, viewport_height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

		GLuint depth;
		glGenRenderbuffers(1, &depth);
		glBindRenderbuffer(GL_RENDERBUFFER, depth);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, viewport_width, viewport_height);

		GLuint fbo;
		glGenFramebuffers(1, &fbo);
		glBindFramebuffer(GL_FRAMEBUFFER, fbo);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);

		glDrawBuffer(GL_COLOR_ATTACHMENT0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glBindFramebuffer(GL_FRAMEBUFFER, fbo);
		glViewport(0, 0, viewport_width, viewport_height);

		const GLint mvp_location = glGetAttribLocation(program, "MVP");
		const GLint vpos_location = glGetAttribLocation(program, "vPos");
		const GLint vcol_location = glGetAttribLocation(program, "vCol");

		GLuint vertex_buffer, color_buffer, index_buffer;
		glGenBuffers(1, &vertex_buffer);
		glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
		glBufferData(GL_ARRAY_BUFFER, 3 * face.getFaceModel().getNumVertices() * sizeof(float), clipspace_vertices_coord.data(), GL_STATIC_DRAW);
		glVertexAttribPointer(vpos_location, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(vpos_location);

		glGenBuffers(1, &color_buffer);
		glBindBuffer(GL_ARRAY_BUFFER, color_buffer);
		glBufferData(GL_ARRAY_BUFFER, 3 * face.getFaceModel().getNumVertices() * sizeof(float), face.calculateColorsDefault().data(), GL_STATIC_DRAW);
		glVertexAttribPointer(vcol_location, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(vcol_location);

		glGenBuffers(1, &index_buffer);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3 * face.getFaceModel().getTriangulation().rows() * sizeof(unsigned int), triangulation.data(), GL_STATIC_DRAW);

		glUseProgram(program);
		glDrawElements(GL_TRIANGLES, 3 * face.getFaceModel().getTriangulation().rows(), GL_UNSIGNED_INT, NULL);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);

		unsigned char* gl_texture_bytes = (unsigned char*)malloc(sizeof(unsigned char) * viewport_height * viewport_width * 3);
		glReadPixels(0, 0, viewport_height, viewport_width, GL_BGR, GL_UNSIGNED_BYTE, gl_texture_bytes);
		cv::Mat img(viewport_height, viewport_width, CV_8UC3, gl_texture_bytes);
		cv::flip(img, img, 0);

		glDeleteFramebuffers(1, &fbo);
		glfwSwapBuffers(window);
		glfwPollEvents();
		return img;
	};

	void terminate_rendering_context() {
		glfwTerminate();
	}

private:
	Renderer() {
		glfwInit();

		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
		window = glfwCreateWindow(720, 720, "face", NULL, NULL);

		glfwMakeContextCurrent(window);

		glClearColor(0.f, 0.f, 0.f, 1.f);
		glClearDepth(1.0f);
		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LEQUAL);
		glShadeModel(GL_SMOOTH);

		glewInit();

		const char* vertex_shader_text =
			"#version 410\n"
			"in vec3 vPos;"
			"in vec3 vCol;"
			"out vec3 color;\n"
			"void main()\n"
			"{\n"
			"    gl_Position = vec4(vPos, 1.0);\n"
			"    color = vCol;\n"
			"}\n";

		const char* fragment_shader_text =
			"#version 410\n"
			"in vec3 color;\n"
			"layout (location = 0) out vec4 fragment;\n"
			"void main()\n"
			"{\n"
			"    fragment = vec4(color, 1);"
			"}\n";

		const GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vertex_shader, 1, &vertex_shader_text, NULL);
		glCompileShader(vertex_shader);

		const GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragment_shader, 1, &fragment_shader_text, NULL);
		glCompileShader(fragment_shader);

		program = glCreateProgram();
		glAttachShader(program, vertex_shader);
		glAttachShader(program, fragment_shader);
		glLinkProgram(program);
		glDeleteShader(vertex_shader);
		glDeleteShader(fragment_shader);
	}

	static Renderer s_instance;
	GLuint program;
	GLFWwindow* window;
};
