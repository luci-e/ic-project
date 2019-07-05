#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>

#include <stdio.h>
#include <string>

#include "shaderHelper.h"
#include "bSimulator.h"
#include "bRenderer.h"

#include "helper_timer.h"
#include "CLI11.hpp"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "bApp.h"

int bApp::initGL()
{
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

	glfwSetWindowUserPointer(window, this);

	glfwSetFramebufferSizeCallback(window, cbReshape);

	/* Make the window's context current */
	glfwMakeContextCurrent(window);

	// start GLEW extension handler
	glewExperimental = GL_TRUE;
	bool err = glewInit() != GLEW_OK;

	if (err)
	{
		fprintf(stderr, "Failed to initialize OpenGL loader!\n");
		return -1;

	}

	// get version info
	const GLubyte* renderer = glGetString(GL_RENDERER); // get renderer string
	const GLubyte* version = glGetString(GL_VERSION); // version as a string
	printf("Renderer: %s\n", renderer);
	printf("OpenGL version supported %s\n", version);

	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	//ImGuiIO& io = ImGui::GetIO(); (void) io;

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();

	const char* glsl_version = "#version 460";

	// Setup Platform/Renderer bindings
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init(glsl_version);

	return 0;
}

int bApp::initSim(int argc, char** argv)
{
	std::vector<int> size = { 256, 256 };
	app.add_option("--size,-s", size, "Size of the simulator")->expected(2);

	app.add_option("--scenario,-i", scenario, "Predefined scenario")->check(CLI::ExistingFile);

	std::vector<std::pair<std::string, COMPUTE_UNIT>> map{
	   {"cpu", COMPUTE_UNIT::CPU}, {"gpu", COMPUTE_UNIT::GPU} };
	app.add_option("--computeUnit,-c", computeUnit, "Device on which to run the simulation")
		->transform(CLI::CheckedTransformer(map, CLI::ignore_case));;

	CLI11_PARSE(app, argc, argv);

	sim = new bSimulator();
	simR = new bRenderer(sim);

	// Initialize the timers for computing time and rendering measurements
	sdkCreateTimer(&timerCompute);
	sdkResetTimer(&timerCompute);

	sdkCreateTimer(&timerGraphics);
	sdkResetTimer(&timerGraphics);

	sim->initSim(size[0], size[1]);
	sim->initNodes();

	simR->initDisplayNodes();

	switch (computeUnit) {
	case COMPUTE_UNIT::CPU: {
		printf("Running on CPU!\n");
		break;
	}

	case COMPUTE_UNIT::GPU: {
		printf("Running on GPU!\n");
		simR->initCudaOpenGLInterop();
		break;
	}
	}

	return 0;
}

void bApp::cbReshape(GLFWwindow* window, int x, int y)
{
	bApp& app = *((bApp*)glfwGetWindowUserPointer(window));

	app.wWidth = x;
	app.wHeight = y;
	glViewport(0, 0, x, y);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, 1, 1, 0, 0, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glfwSwapBuffers(window);
}

void bApp::start()
{
	status = SIM_STATUS::RUNNING;
	ImVec4 clear_color = ImVec4(0, 0, 0, 1.00f);

	while (!glfwWindowShouldClose(window)) {
		switch (status) {
		case SIM_STATUS::RUNNING: {
			// wipe the drawing surface clear
			glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

			// Start the Dear ImGui frame
			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();

			update();
			simR->render();

			if (ImGui::BeginMainMenuBar())
			{


				if (ImGui::BeginMenu("File"))
				{
					if (ImGui::MenuItem("Quit", "Alt+F4")) {
						glfwSetWindowShouldClose(window, GLFW_TRUE);
					}
					ImGui::EndMenu();
				}


				if (ImGui::BeginMenu("View"))
				{
					ImGui::Checkbox("Stats Window", &showStatsWindow);
					ImGui::Checkbox("Tools Window", &showToolsWindow);
					ImGui::EndMenu();
				}

				ImGui::EndMainMenuBar();
			}

			if (showStatsWindow) {
				if (ImGui::Begin("Stats", &showStatsWindow)) {
					ImGui::Text("Application average compute time %.3f ms/frame (%.1f SPS)", averageComputeTime, 1000.f / averageComputeTime );
					ImGui::Text("Application average render time %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
					ImGui::End();
				}
			}

			// Rendering
			ImGui::Render();
			int display_w, display_h;
			glfwGetFramebufferSize(window, &display_w, &display_h);
			glViewport(0, 0, display_w, display_h);

			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
			// put the stuff we've been drawing onto the display
			glfwSwapBuffers(window);
			break;
		}

		case SIM_STATUS::PAUSED: {
			break;
		}
		}

		// update other events like input handling 
		glfwPollEvents();
	}
}

void bApp::update()
{
	switch (computeUnit) {
	case COMPUTE_UNIT::CPU: {
		sdkStartTimer(&timerCompute);
		sim->CPUUpdate();
		sdkStopTimer(&timerCompute);
		averageComputeTime = sdkGetAverageTimerValue(&timerCompute);

		sdkStartTimer(&timerGraphics);
		simR->CPUUpdateGraphics();
		sdkStopTimer(&timerGraphics);

		break;
	}

	case COMPUTE_UNIT::GPU: {
		sdkStartTimer(&timerCompute);
		sim->GPUUpdate();
		sdkStopTimer(&timerCompute);
		averageComputeTime = sdkGetAverageTimerValue(&timerCompute);

		sdkStartTimer(&timerGraphics);
		simR->GPUUpdateGraphics();
		sdkStopTimer(&timerGraphics);

		break;
	}
	}
}

void bApp::cleanup()
{
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(window);
	glfwTerminate();

	sim->cleanup();
	cudaFree(sim);
}
