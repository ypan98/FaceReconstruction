//#pragma once
#define GLEW_STATIC

#include "Eigen.h"
#include "FaceModel.h"
#include "opencv2/core/core.hpp"
#include <GL/glew.h>
#include <GLFW/glfw3.h>

/*
	In order to initialize this render class you have to specify the size of the viewport and also the face model
	you are using.
*/
class Renderer{
public:
	Renderer(int height, int width, FaceModel face_model) {
		viewport_height = height;
		viewport_width = width;

		glfwInit();

		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
		window = glfwCreateWindow(viewport_width, viewport_height, "face", NULL, NULL);

		glfwMakeContextCurrent(window);

		glClearColor(0.f, 0.f, 0.f, 1.f);
		glClearDepth(1.0f);
		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LEQUAL);
		glShadeModel(GL_SMOOTH);

		glewInit();

		const char* vertex_shader_text =
			"#version 140\n"
			"#extension GL_ARB_explicit_attrib_location : require\n"
			"uniform mat4 proj_matrix;"
			"in vec3 vPos;"
			"in vec3 vCol;"
			"out vec3 color;\n"
			"void main()\n"
			"{\n"
			"    gl_Position = proj_matrix * vec4(vPos, 1.0);\n"
			"    color = vCol;\n"
			"}\n";

		const char* fragment_shader_text =
			"#version 140\n"
			"#extension GL_ARB_explicit_attrib_location : require\n"
			"in vec3 color;\n"
			"layout (location = 0) out vec4 fragment;\n"
			"void main()\n"
			"{\n"
			"    fragment = vec4(color, 1);"
			"}\n";

		GLint isCompiled = 0;

		const GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vertex_shader, 1, &vertex_shader_text, NULL);
		glCompileShader(vertex_shader);
		glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &isCompiled);
		if (isCompiled == GL_FALSE)
		{
			GLint maxLength = 0;
			glGetShaderiv(vertex_shader, GL_INFO_LOG_LENGTH, &maxLength);

			// The maxLength includes the NULL character
			std::vector<GLchar> errorLog(maxLength);
			glGetShaderInfoLog(vertex_shader, maxLength, &maxLength, &errorLog[0]);

			for (int i = 0; i < errorLog.size(); i++) {
				std::cout << errorLog[i];
			}
			std::cout << std::endl;
		}

		const GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragment_shader, 1, &fragment_shader_text, NULL);
		glCompileShader(fragment_shader);
		glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &isCompiled);
		if (isCompiled == GL_FALSE)
		{
			GLint maxLength = 0;
			glGetShaderiv(fragment_shader, GL_INFO_LOG_LENGTH, &maxLength);

			// The maxLength includes the NULL character
			std::vector<GLchar> errorLog(maxLength);
			glGetShaderInfoLog(fragment_shader, maxLength, &maxLength, &errorLog[0]);

			for (int i = 0; i < errorLog.size(); i++) {
				std::cout << errorLog[i];
			}
			std::cout << std::endl;
		}

		program = glCreateProgram();
		glAttachShader(program, vertex_shader);
		glAttachShader(program, fragment_shader);
		glLinkProgram(program);
		glDeleteShader(vertex_shader);
		glDeleteShader(fragment_shader);

		//Bind FBO
		glGenTextures(1, &color);
		glBindTexture(GL_TEXTURE_2D, color);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, viewport_width, viewport_height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

		glGenRenderbuffers(1, &depth);
		glBindRenderbuffer(GL_RENDERBUFFER, depth);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, viewport_width, viewport_height);

		glGenFramebuffers(1, &fbo);
		glBindFramebuffer(GL_FRAMEBUFFER, fbo);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);

		glDrawBuffer(GL_COLOR_ATTACHMENT0);

		const GLint vpos_location = glGetAttribLocation(program, "vPos");
		const GLint vcol_location = glGetAttribLocation(program, "vCol");

		//Bind VBOs
		Eigen::Matrix<unsigned int, -1, -1> triangulation = face_model.getTriangulation().transpose().cast <unsigned int>();

		glGenBuffers(1, &vertex_buffer);
		glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
		glBufferData(GL_ARRAY_BUFFER, 3 * face_model.getNumVertices() * sizeof(float), face_model.getShapeMean().data(), GL_STATIC_DRAW);

		glVertexAttribPointer(vpos_location, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(vpos_location);

		glGenBuffers(1, &color_buffer);
		glBindBuffer(GL_ARRAY_BUFFER, color_buffer);
		glBufferData(GL_ARRAY_BUFFER, 3 * face_model.getNumVertices() * sizeof(float), face_model.getColorMean().data(), GL_STATIC_DRAW);
		glVertexAttribPointer(vcol_location, 3, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(vcol_location);

		glGenBuffers(1, &index_buffer);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3 * face_model.getTriangulation().rows() * sizeof(unsigned int), triangulation.data(), GL_STATIC_DRAW);

		glUseProgram(program);
		GLint render_projection_matrix_loc = glGetUniformLocation(program, "proj_matrix");
	}

	std::pair<cv::Mat, cv::Mat> render(Face face, Matrix4f projection_matrix, bool enable_backface_culling = true, bool enable_near_clipping = true, bool enable_far_clipping = true)
	{
		VectorXf vertices = face.calculateVerticesDefault();
		VectorXf colors = face.calculateColorsDefault();

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glViewport(0, 0, viewport_width, viewport_height);

		glUniformMatrix4fv(render_projection_matrix_loc, 1, GL_FALSE, projection_matrix.data());

		glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
		glNamedBufferSubData(vertex_buffer, 0, 3 * face.getFaceModel().getNumVertices() * sizeof(float), vertices.data());

		glBindBuffer(GL_ARRAY_BUFFER, color_buffer);
		glNamedBufferSubData(color_buffer, 0, 3 * face.getFaceModel().getNumVertices() * sizeof(float), colors.data());
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);

		glDrawElements(GL_TRIANGLES, 3 * face.getFaceModel().getTriangulation().rows(), GL_UNSIGNED_INT, NULL);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		unsigned char* gl_texture_bytes = (unsigned char*)malloc(sizeof(unsigned char) * viewport_height * viewport_width * 3);
		float* gl_depth_floats = (float*)malloc(sizeof(float) * viewport_height * viewport_width);
		glReadPixels(0, 0, viewport_height, viewport_width, GL_BGR, GL_UNSIGNED_BYTE, gl_texture_bytes);
		glReadPixels(0, 0, viewport_height, viewport_width, GL_DEPTH_COMPONENT, GL_FLOAT, gl_depth_floats);
		cv::Mat img(viewport_height, viewport_width, CV_8UC3, gl_texture_bytes);
		cv::Mat depth(viewport_height, viewport_width, CV_32FC1, gl_depth_floats);
		cv::flip(img, img, 0);
		cv::flip(depth, depth, 0);
		glfwSwapBuffers(window);
		glfwPollEvents();

		return std::make_pair(img, depth);
	};

	void terminate_rendering_context() {
		glfwTerminate();
	}

private:
	int viewport_width, viewport_height;
	GLuint program, vertex_buffer, color_buffer, index_buffer, color, depth, fbo, render_projection_matrix_loc;
	GLFWwindow* window;
};
