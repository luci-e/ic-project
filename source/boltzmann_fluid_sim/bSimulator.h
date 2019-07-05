#ifndef __B_SIMULATOR__
#define __B_SIMULATOR__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>
#include "bCommon.h"

class bSimulator : public Managed
{
public:
	enum edgeBehaviour {
		LOOP,
		EXIT
	} doAtEdge = edgeBehaviour::LOOP;

	float weights[9] = { 4.f / 9.f,
						1.f / 9.f,
						1.f / 9.f,
						1.f / 9.f,
						1.f / 9.f,
						1.f / 36.f,
						1.f / 36.f,
						1.f / 36.f,
						1.f / 36.f
	};

	float speeds[9 * 2] = { 0.f, 0.f,
							1.f, 0.f,
							0.f, -1.f,
							-1.f, 0.f,
							0.f, 1.f,
							1.f, -1.f,
							-1.f, -1.f,
							-1.f, 1.f,
							1.f, 1.f
	};

	int directions[9][2] = { {0, 0},
							{1, 0},
							{0, -1},
							{-1, 0},
							{0, 1},
							{1, -1},
							{-1, -1},
							{-1, 1},
							{1, 1}
	};

	dim3 gridDim;
	dim3 blockDim = { 16, 16 };
	unsigned long long int dimX = 16, dimY = 16;

	double temperature = 1;
	double mass = 2.992e-23;
	const double boltzmann_k = 1.38064852e-23;
	double c = 0;
	double csqr = 0;
	double viscosity = 1.0;

	unsigned long long int totalPoints = 0;

	// Particle data
	node* nodes = NULL;

	bSimulator() {};

	void initSim(unsigned long long int dimX = 256, unsigned long long int dimY = 256) {
		totalPoints = dimX * dimY;
		csqr = 3.f * boltzmann_k * temperature / mass;
		csqr = 1;
		printf("Csqr is :%lf\n", csqr);
		c = sqrt(csqr);

		this->dimX = dimX;
		this->dimY = dimY;

		gridDim = {
			(unsigned int) ceil((double)dimX / (double)blockDim.x),
			(unsigned int) ceil((double)dimY / (double)blockDim.y)
		};

		printf("Grid dim: %u %u \nBlock dim: %u %u\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);
	};

	void CPUUpdate();
	void CPUComputeVelocity();
	void CPUcomputeEquilibrium();
	void CPUcomputeNew();
	void CPUstream();

	void GPUUpdate();
	void GPUComputeVelocity();
	void GPUcomputeEquilibrium();
	void GPUcomputeNew();
	void GPUstream();

	int initNodes();
	bool inside(long long int x, long long int y);

	void cleanup();
};

#endif // !__B_SIMULATOR__
