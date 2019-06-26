#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>

class bSimulator
{
public:

	struct particle {
		float x, y;
		float r,g,b;
	};

	unsigned int dimX = 256, dimY = 256;
	unsigned int totalPoints;
	struct cudaGraphicsResource* cuda_vbo_resource; // handles OpenGL-CUDA exchange

	// Particle data
	GLuint vbo = 0;                 // OpenGL vertex buffer object
	GLuint vao = 0;                 // OpenGL vertex buffer object

	bSimulator(){
		totalPoints = dimX * dimY;
	};

	bSimulator(unsigned int dimX, unsigned int dimY) : dimX(dimX), dimY(dimY) {
		totalPoints = dimX * dimY;
	};

	void InitParticles();
};