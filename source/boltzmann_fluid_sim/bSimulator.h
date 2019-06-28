#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>

#define SQRT2 1.41421356237

class bSimulator
{
public:

	struct particle {
		float2 pos;
		float3 col;
	};

	enum class nodeType : unsigned int {
		BASE,
		WALL
	};

	struct displayNode {
		float2 pos;
		float2 vel;
		float density;
		nodeType ntype;
	};

	struct node {
		nodeType ntype;
		long long int x, y;
		float densities[9];
		float eqDensities[9];

		float2 vel;
	};

	unsigned long long int dimX = 256, dimY = 256;
	const float weights[9] = { 4.f / 9.f,
							   1.f / 9.f,
							   1.f / 9.f,
							   1.f / 9.f,
							   1.f / 9.f,
							   1.f / 36.f,
							   1.f / 36.f,
							   1.f / 36.f,
							   1.f / 36.f
	};

	const float speeds[9 * 2] = { 0.f, 0.f,
								1.f, 0.f,
								0.f, -1.f,
								-1.f, 0.f,
								0.f, 1.f,
								1.f, -1.f,
								-1.f, -1.f,
								-1.f, 1.f,
								1.f, 1.f
	};

	const int directions[9][2] = { {0, 0},
								   {1, 0},
								   {0, -1},
								   {-1, 0},
								   {0, 1},
								   {1, -1},
								   {-1, -1},
								   {-1, 1},
								   {1, 1}
	};

	double temperature = 273.0;
	double mass = 1;
	const double boltzmann_k = 1.38064852e-23;
	double c = 0;
	double csqr = 0;
	double viscosity = 1;

	unsigned long long int totalPoints = 0;
	struct cudaGraphicsResource* cuda_vbo_resource = NULL; // handles OpenGL-CUDA exchange

	// Particle data
	node* nodes;
	GLuint nodesBuffer = 0;                 // OpenGL vertex buffer object
	GLuint vao = 0;                 // OpenGL vertex buffer object

	bSimulator(unsigned long long int dimX = 256, unsigned long long int dimY = 256) : dimX(dimX), dimY(dimY) {
		totalPoints = dimX * dimY;
		csqr = 3 * boltzmann_k * temperature / mass;
		csqr = 1;
		c = sqrt(csqr);
	};

	void InitParticles();

	void CPUUpdate();
	void CPUcomputeEquilibrium();
	void CPUcomputeNew();
	void CPUstream();

	int initNodes(float density);
	int initDisplayNodes();
	int updateDisplayNodes();
	void initDisplayNode(const node& n, displayNode& dn);
	void updateDisplayNode(const node& n, displayNode& dn);

	inline bool inside(long long int x, long long int y);
};