
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>
#include <chrono>
#include <thread>

#include "shaderHelper.h"
#include "bSimulator.h"

GLFWwindow* window;

bSimulator* sim;

// Window properties
int wWidth = 0, wHeight = 0;

void cb_reshape(GLFWwindow* window, int x, int y)
{
	wWidth = x;
	wHeight = y;
	glViewport(0, 0, x, y);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, 1, 1, 0, 0, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glfwSwapBuffers(window);
}


int initGL() {
	/* Initialize the library */
	if (!glfwInit())
		return -1;

	/* Create a windowed mode window and its OpenGL context */
	window = glfwCreateWindow(640, 480, "Boltzmann Simulation", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		return -1;
	}

	glfwSetFramebufferSizeCallback(window, cb_reshape);

	/* Make the window's context current */
	glfwMakeContextCurrent(window);

	// start GLEW extension handler
	glewExperimental = GL_TRUE;
	glewInit();

	return 0;
}

void cleanup() {
	sim->cleanup();
	cudaFree(sim);
}


int main()
{
	if (int r = initGL() < 0) {
		return r;
	}

	// get version info
	const GLubyte* renderer = glGetString(GL_RENDERER); // get renderer string
	const GLubyte* version = glGetString(GL_VERSION); // version as a string
	printf("Renderer: %s\n", renderer);
	printf("OpenGL version supported %s\n", version);

	//------------------------------------------------//

	sim = new bSimulator();

	Shader shd("vertex.glsl", "fragment.glsl");
	shd.use();
	sim->initSim(256, 256);
	sim->initNodes();

	// wipe the drawing surface clear
	glClearColor(0, 0, 0, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glBindVertexArray(sim->vao);
	glDrawArrays(GL_POINTS, 0, sim->totalPoints);
	// put the stuff we've been drawing onto the display
	glfwSwapBuffers(window);

	while (!glfwWindowShouldClose(window)) {
		sim->GPUUpdate();
		// wipe the drawing surface clear
		glClearColor(0,0,0, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glBindVertexArray(sim->vao);
		glDrawArrays(GL_POINTS, 0, sim->totalPoints);
		// put the stuff we've been drawing onto the display
		glfwSwapBuffers(window);
		// update other events like input handling 
		glfwPollEvents();		
	}

	//------------------------------------------------//

	cleanup();

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	// close GL context and any other GLFW resources
	glfwTerminate();

	return 0;
}