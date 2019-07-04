#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>

#include <stdio.h>
#include <string>

#include "shaderHelper.h"
#include "bSimulator.h"
#include "helper_timer.h"
#include "CLI11.hpp"
class bApp
{
public:

	enum class COMPUTE_UNIT {
		CPU = 0,
		GPU = 1
	} computeUnit = COMPUTE_UNIT::GPU;

	enum class SIM_STATUS {
		STOPPED = 0,
		RUNNING = 1,
		PAUSED = 2
	} status = SIM_STATUS::PAUSED;


	CLI::App app{ "Lattice Boltzmann Fluid Sim" };
	Shader shd;

	GLFWwindow* window;

	bSimulator* sim;

	StopWatchInterface* timerCompute = NULL;
	StopWatchInterface* timerGraphics = NULL;
	float averageComputeTime = 1;

	bool showStatsWindow = false;
	bool showToolsWindow = false;

	// Window properties
	int wWidth = 0, wHeight = 0;

	std::string scenario = "";

	bApp() {};

	int initGL();
	int initSim(int argc, char** argv);
	
	static void cbReshape(GLFWwindow* window, int x, int y);

	void start();
	void update();

	void cleanup();
};

