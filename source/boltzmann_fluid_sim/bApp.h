#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>

#include <stdio.h>
#include <string>
#include <map>

#include "bSimulator.h"
#include "bRenderer.h"

#include "utilities.h"
#include "imgui.h"
#include "helper_timer.h"
#include "CLI11.hpp"

class bApp
{
public:

	class tool {
	public:
		int brushSize = 1;
		float startTime = 0.f;
		float startX = 0.f, startY = 0.f;
		float deltaX = 0.f, deltaY = 0.f;
		bool* brush = nullptr;

		bApp* app;

		tool(bApp* pApp) {
			app = pApp;
			generateBrush();
		};



		virtual void handleMovement() = 0;
		virtual void handleClick() = 0;
		virtual void handleInput(ImGuiIO* io) = 0;

		void scaleCoordinates() {
			float scaleX = (float)app->sim->dimX / app->wWidth;
			float scaleY = (float)app->sim->dimY / app->wHeight;

			startX = mapNumber<float>(startX, 0, app->wWidth, 0, app->sim->dimX);
			startY = mapNumber<float>(startY, 0, app->wHeight, 0, app->sim->dimY);
			deltaX *= scaleX;
			deltaY *= scaleY;
		}

		virtual void updateGraphics() {
			if (ImGui::SliderInt("Brush Size", &brushSize, 1, 100)) {
				generateBrush();
			}
		};


		// Generate a circular brush of the diameter brushSize
		void generateBrush() {
			free(brush);
			brush = (bool*)malloc(sizeof(bool) * pow(brushSize, 2));
			memset(brush, false, sizeof(bool) * pow(brushSize, 2));

			float centreX = (float)brushSize / 2.f;
			float centreY = (float)brushSize / 2.f;


			for (auto y = 0; y < brushSize; y++) {
				for (auto x = 0; x < brushSize; x++) {
					float distance = sqrt(pow((float)x - centreX, 2) + pow((float)y - centreY, 2));
					if (distance <= (float)brushSize / 2.f) {
						*(brush + y * brushSize + x) = true;
					}
				}
			}
		}

	};

	class moveTool : public tool {
	public:
		using tool::tool;

		void handleMovement() {};
		void handleClick() {};

		void updateGraphics() {
			tool::updateGraphics();
		};

		void handleInput(ImGuiIO* io) {};
	};

	class fanTool : public tool {
	public:
		using tool::tool;

		void handleMovement() {};
		void handleClick() {};

		void updateGraphics() {
			tool::updateGraphics();

		};

		void handleInput(ImGuiIO* io) {};

	};

	class sourceSinkTool : public tool {
	public:
		using tool::tool;
		bool sourceOrSink = true;

		void drawSourceSink(float startX, float startY) {
			for (auto y = 0; y < brushSize; y++) {
				for (auto x = 0; x < brushSize; x++) {
					bool isActive = *(brush + y * brushSize + x);

					if (isActive) {

						int dX = x - brushSize / 2;
						int dY = y - brushSize / 2;

						int newX = startX + dX;
						int newY = startY + dY;
						
						if (app->sim->inside(newX, newY)) {
							node& n = *(app->sim->nodes + app->sim->dimX * newY + newX);
							if (sourceOrSink) {
								n.ntype = nodeType::SOURCE;
								memset(n.densities, 1.f, 9 * sizeof(float));
							}
							else {
								n.ntype = nodeType::SINK;
							}
						}

					}

				}
			}
		};

		void handleMovement() {
			scaleCoordinates();
			cudaDeviceSynchronize();

			float direction[2] = { deltaX, deltaY };
			float mag = magnitude(direction, 2);
			normalize(direction, direction, 2);

			for (float step = 0; step < mag; step++) {
				float dX = step * direction[0];
				float dY = step * direction[1];


				int newX = startX + dX;
				int newY = startY + dY;

				if (app->sim->inside(newX, newY)) {
					drawSourceSink(newX, newY);
				}
			}
		};

		void handleClick() {
			scaleCoordinates();
			cudaDeviceSynchronize();
			drawSourceSink(startX, startY);
		};

		void updateGraphics() {
			tool::updateGraphics();
			if (ImGui::Button(sourceOrSink ? "Source" : "Sink")) {
				sourceOrSink = !sourceOrSink;
			}
		};

		void handleInput(ImGuiIO* io) {
			if (ImGui::IsMouseClicked(0)) {
				startX = io->MousePos.x;
				startY = io->MousePos.y;
				handleClick();
			}
			else if (ImGui::IsMouseClicked(1)) {
				startX = io->MousePos.x;
				startY = io->MousePos.y;
			}

			if (ImGui::IsMouseDown(0)) {

				deltaX = io->MouseDelta.x;
				deltaY = io->MouseDelta.y;

				startX = io->MousePos.x;
				startY = io->MousePos.y;
				handleMovement();
			}
			else if (ImGui::IsMouseDown(1)) {

			}
		};

	};

	class wallTool : public tool {
	public:
		using tool::tool;

		void drawWall(float startX, float startY) {
			for (auto y = 0; y < brushSize; y++) {
				for (auto x = 0; x < brushSize; x++) {
					bool isActive = *(brush + y * brushSize + x);

					if (isActive) {

						int dX = x - brushSize / 2;
						int dY = y - brushSize / 2;

						int newX = startX + dX;
						int newY = startY + dY;


						if (app->sim->inside(newX, newY)) {
							node& n = *(app->sim->nodes + app->sim->dimX * newY + newX);
							n.ntype = nodeType::WALL;
						}
						
					}

				}
			}
		};

		void handleMovement() {
			scaleCoordinates();
			cudaDeviceSynchronize();

			float direction[2] = { deltaX, deltaY };
			float mag = magnitude(direction, 2);
			normalize(direction, direction, 2);

			for (float step = 0; step < mag; step++) {
				float dX = step * direction[0];
				float dY = step * direction[1];


				int newX = startX + dX;
				int newY = startY + dY;

				if (app->sim->inside(newX, newY)) {
					drawWall(newX, newY);
				}
			}
			
		};

		void handleClick() {
			scaleCoordinates();
			cudaDeviceSynchronize();
			drawWall(startX, startY);
		};

		void updateGraphics() {
			tool::updateGraphics();
		};

		void handleInput(ImGuiIO* io) {

			if (ImGui::IsMouseClicked(0)) {
				startX = io->MousePos.x;
				startY = io->MousePos.y;
				handleClick();
			}
			else if (ImGui::IsMouseClicked(1)) {
				startX = io->MousePos.x;
				startY = io->MousePos.y;
			}

			if (ImGui::IsMouseDown(0)) {

				deltaX = io->MouseDelta.x;
				deltaY = io->MouseDelta.y;

				startX = io->MousePos.x;
				startY = io->MousePos.y;
				handleMovement();
			}
			else if (ImGui::IsMouseDown(1)) {

			}
		};
	};

	enum class COMPUTE_UNIT {
		CPU = 0,
		GPU = 1
	} computeUnit = COMPUTE_UNIT::GPU;

	std::map<COMPUTE_UNIT, char*> computeString = { {COMPUTE_UNIT::CPU, "CPU"}, {COMPUTE_UNIT::GPU, "GPU"} };

	enum class SIM_STATUS {
		STOPPED = 0,
		RUNNING = 1,
		PAUSED = 2
	} status = SIM_STATUS::PAUSED;

	std::map< SIM_STATUS, std::string> statusString = { {SIM_STATUS::STOPPED, "Stopped"},
	{SIM_STATUS::RUNNING, "Running"}, {SIM_STATUS::PAUSED, "Paused"} };


	CLI::App app{ "Lattice Boltzmann Fluid Sim" };

	GLFWwindow* window;

	bSimulator* sim;
	bRenderer* simR;

	StopWatchInterface* timerCompute = NULL;
	StopWatchInterface* timerGraphics = NULL;
	float averageComputeTime = 1;

	bool showInitializerWindow = false;
	bool showStatsWindow = false;
	bool showControlWindow = false;

	int edgeBehaviour = 0;
	int renderMode = 0;



	int selectedTool = -1;
	std::map< int, tool*> tools = { {0, new moveTool(this)}, {1, new fanTool(this)}, {2, new sourceSinkTool(this)}, {3, new wallTool(this)} };

	// Window properties
	int wWidth = 640, wHeight = 480;

	std::string scenario = "";

	bApp() {};

	int initGL();
	int initSim(int argc, char** argv);

	static void cbReshape(GLFWwindow* window, int x, int y);

	void start();
	void updateSim();

	void updateGraphics();

	void updateRenderMode();
	void updateEdgeBehaviour();

	void cleanup();
};

