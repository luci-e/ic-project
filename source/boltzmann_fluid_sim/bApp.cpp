#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>

#include <stdio.h>
#include <string>
#include <algorithm>    // std::find

#include "utilities.h"
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
	simR->initCudaOpenGLInterop();

	switch (computeUnit) {
	case COMPUTE_UNIT::CPU: {
		printf("Running on CPU!\n");
		break;
	}

	case COMPUTE_UNIT::GPU: {
		printf("Running on GPU!\n");
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
	ImVec4 clear_color = ImVec4(0.f, 0.f, 0.f, 1.00f);
	ImGuiIO& io = ImGui::GetIO(); (void)io;

	updateSim();
	updateGraphics();

	while (!glfwWindowShouldClose(window)) {
		// update other events like input handling 
		glfwPollEvents();

		// wipe the drawing surface clear
		glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

		// Start the Dear ImGui frame
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		if (status == SIM_STATUS::RUNNING) {
			updateSim();
			updateGraphics();
		}
		simR->render();

		if (ImGui::BeginMainMenuBar())
		{
			if (ImGui::BeginMenu("File"))
			{
				if (ImGui::MenuItem("New Simulator")) {
					showInitializerWindow = true;
				}

				if (ImGui::MenuItem("Quit", "Alt+F4")) {
					glfwSetWindowShouldClose(window, GLFW_TRUE);
				}

				ImGui::EndMenu();
			}


			if (ImGui::BeginMenu("View"))
			{
				ImGui::Checkbox("Stats Window", &showStatsWindow);
				ImGui::Checkbox("Tools Window", &showControlWindow);
				ImGui::EndMenu();
			}

			ImGui::EndMainMenuBar();
		}

		// Stats window displays FPS and SPS
		if (showStatsWindow) {
			ImGui::Begin("Stats", &showStatsWindow);
			if (status == SIM_STATUS::PAUSED) {
				ImGui::Text("Current compute unit:");
				ImGui::SameLine();
				if (ImGui::Button(computeString[computeUnit])) {
					timerCompute->reset();
					cudaDeviceSynchronize();

					switch (computeUnit) {
					case COMPUTE_UNIT::CPU:
						computeUnit = COMPUTE_UNIT::GPU;
						break;

					case COMPUTE_UNIT::GPU:
						computeUnit = COMPUTE_UNIT::CPU;
						break;
					}

				}
			}
			else {
				ImGui::Text("Current compute unit: %s", computeString[computeUnit]);
			}

			ImGui::Text("Application average compute time %.3f ms/frame (%.1f SPS)", averageComputeTime, 1000.f / averageComputeTime);
			ImGui::Text("Application average render time %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
			ImGui::End();
		}


		// Control window for the simulator
		if (showControlWindow) {
			ImGui::Begin("Simulator", &showControlWindow);

			ImGui::Text("Simulator Status:");
			ImGui::SameLine();
			if (ImGui::Button(statusString[status].c_str())) {
				if (status == SIM_STATUS::PAUSED) {
					status = SIM_STATUS::RUNNING;
				}
				else if (status == SIM_STATUS::RUNNING) {
					status = SIM_STATUS::PAUSED;
				}
			}

			if (status == SIM_STATUS::PAUSED) {
				ImGui::SameLine();
				ImGui::PushButtonRepeat(true);
				if (ImGui::Button("Step") ){
					updateSim();
					updateGraphics();
				}
				ImGui::PopButtonRepeat();
			}

			if (status == SIM_STATUS::PAUSED) {
				ImGui::SameLine();
				if (ImGui::Button("Reset")) {
					sim->reset();
					updateSim();
					updateGraphics();
				}
			}
			if (ImGui::TreeNode("Simulation parameters"))
			{
				ImGui::SliderFloat("Viscosity", &sim->viscosity, 0.0f, 2.0f, "%.4f");
				ImGui::SliderFloat("C", &sim->c, 0.0f, 1.0f, "%.4f");

				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Edge Behaviour"))
			{
				ImGui::RadioButton("Loop", &edgeBehaviour, 0);
				ImGui::RadioButton("Exit", &edgeBehaviour, 1);
				ImGui::RadioButton("Wall", &edgeBehaviour, 2);
				ImGui::TreePop();
			}

			if (ImGui::TreeNode("Render Mode"))
			{
				ImGui::RadioButton("Mesh", &renderMode, 0);
				ImGui::RadioButton("Points", &renderMode, 1);
				ImGui::TreePop();

			}

			if (ImGui::IsMousePosValid()) {
				if (!io.WantCaptureMouse) {
					if (selectedTool >= 0) {
						tools[selectedTool]->handleInput(&io);
						updateGraphics();
					}
				}
			}

			if (ImGui::TreeNode("Tools")) {
				if (ImGui::Selectable("Move Tool", selectedTool == 0)) {
					selectedTool = 0;
				}

				if (ImGui::Selectable("Fan Tool", selectedTool == 1)) {
					selectedTool = 1;
				}

				if (ImGui::Selectable("Source Sink Tool", selectedTool == 2)) {
					selectedTool = 2;
				}

				if (ImGui::Selectable("Wall Tool", selectedTool == 3)) {
					selectedTool = 3;
				}

				if (selectedTool >= 0) {
					tools[selectedTool]->updateGraphics();
				}
				ImGui::TreePop();
			}

			ImGui::End();
		}

		updateRenderMode();
		updateEdgeBehaviour();


		if (showInitializerWindow) {
			static int newSize[2] = { 256, 256 };

			ImGui::Begin("New Simulation", &showInitializerWindow);
			ImGui::InputInt2("Size", newSize);

			if (ImGui::Button("Simulate!")) {
				cudaDeviceSynchronize();
				simR->cleanup();
				sim->cleanup();
				cudaFree(sim);
				cudaDeviceSynchronize();

				bSimulator* newSim = new bSimulator();
				newSim->initSim(newSize[0], newSize[1]);
				newSim->initNodes();
				simR->sim = newSim;
				simR->initDisplayNodes();
				simR->initCudaOpenGLInterop();
			}

			ImGui::End();
		}

		// Rendering
		ImGui::Render();
		int display_w, display_h;
		glfwGetFramebufferSize(window, &display_w, &display_h);
		glViewport(0, 0, display_w, display_h);

		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		// put the stuff we've been drawing onto the display
		glfwSwapBuffers(window);
	}
}

void bApp::updateSim()
{
	switch (computeUnit) {
	case COMPUTE_UNIT::CPU: {
		sdkStartTimer(&timerCompute);
		sim->CPUUpdate();
		sdkStopTimer(&timerCompute);
		averageComputeTime = sdkGetAverageTimerValue(&timerCompute);
		break;
	}

	case COMPUTE_UNIT::GPU: {
		sdkStartTimer(&timerCompute);
		sim->GPUUpdate();
		sdkStopTimer(&timerCompute);
		averageComputeTime = sdkGetAverageTimerValue(&timerCompute);

		break;
	}
	}
}

void bApp::updateGraphics()
{
	switch (computeUnit) {
	case COMPUTE_UNIT::CPU: {
		sdkStartTimer(&timerGraphics);
		simR->CPUUpdateGraphics();
		sdkStopTimer(&timerGraphics);

		break;
	}

	case COMPUTE_UNIT::GPU: {
		sdkStartTimer(&timerGraphics);
		simR->GPUUpdateGraphics();
		sdkStopTimer(&timerGraphics);

		break;
	}
	}
}

void bApp::updateRenderMode()
{
	bRenderer::renderMode rm = bRenderer::renderMode(renderMode);

	if (rm != simR->renderM) {
		simR->setRenderMode(rm);
	}
}

void bApp::updateEdgeBehaviour()
{
	bSimulator::edgeBehaviour eb = bSimulator::edgeBehaviour(edgeBehaviour);
	if (eb != sim->doAtEdge) {
		sim->setEdgeBehaviour(eb);
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
