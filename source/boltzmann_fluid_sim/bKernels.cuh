#ifndef __BOLTZMANN_KERNELS_CUH_
#define __BOLTZMANN_KERNELS_CUH_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "bRenderer.h"
#include "bSimulator.h"

__global__ void
cudaComputeVelocity(bSimulator* sim);

__global__ void
cudaComputeEquilibrium(bSimulator* sim);

__global__ void
cudaComputeNew(bSimulator* sim);

__global__ void
cudaStream(bSimulator* sim);

__global__ void
cudaUpdateGraphics(bRenderer* simR);

extern "C" {
	void computeVelocity(bSimulator* sim);
	void computeEquilibrium(bSimulator* sim);
	void computeNew(bSimulator* sim);
	void stream(bSimulator* sim);
	void updateGraphics(bRenderer* simR);
}

#endif