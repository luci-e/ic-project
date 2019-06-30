#ifndef __B_SIMULATOR__
#define __B_SIMULATOR__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>


#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#define SQRT2 1.41421356237

class Managed {
public:
	void* operator new(size_t len) {
		void* ptr;
		cudaMallocManaged(&ptr, len);
		cudaDeviceSynchronize();
		return ptr;
	}

	void operator delete(void* ptr) {
		cudaDeviceSynchronize();
		cudaFree(ptr);
	}
};


class bSimulator : public Managed
{
public:

	enum edgeBehaviour{
		LOOP,
		EXIT
	};

	enum nodeType{
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
		float newDensities[9];
		float eqDensities[9];

		float2 vel;
		float2 addVel;

	};

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
	unsigned long long int dimX = 256, dimY = 256;
	edgeBehaviour doAtEdge = edgeBehaviour::LOOP;

	double temperature = 1;
	double mass = 2.992e-23;
	const double boltzmann_k = 1.38064852e-23;
	double c = 0;
	double csqr = 0;
	double viscosity = 1.0;
	float testVar;

	unsigned long long int totalPoints = 0;
	struct cudaGraphicsResource* cuda_vbo_resource = NULL; // handles OpenGL-CUDA exchange

	// Particle data
	node* nodes = NULL;
	GLuint nodesBuffer = 0;                 // OpenGL vertex buffer object
	GLuint vao = 0;                 // OpenGL vertex buffer object

	bSimulator() {};

	void initSim(unsigned long long int dimX = 256, unsigned long long int dimY = 256) {
		totalPoints = dimX * dimY;
		csqr = 3.f * boltzmann_k * temperature / mass;
		printf("Csqr is :%lf", csqr);
		c = sqrt(csqr);

		this->dimX = dimX;
		this->dimY = dimY;

		gridDim = {
			(unsigned int)ceil((double)dimX / (double)blockDim.x),
			(unsigned int)ceil((double)dimY / (double)blockDim.y),
			1
		};
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
	int initDisplayNodes();
	int updateDisplayNodes();
	void initDisplayNode(const node& n, displayNode& dn);
	void updateDisplayNode(const node& n, displayNode& dn);

	bool inside(long long int x, long long int y);

	void cleanup();
};

#endif // !__B_SIMULATOR__
