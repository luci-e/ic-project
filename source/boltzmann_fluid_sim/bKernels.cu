#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "bKernels.cuh"
#include "utilities.h"
#include "bSimulator.h"

__device__ inline bool inside(long long int x, long long int y, long long int maxX, long long int maxY){
	return (x >= 0 && x < maxX && y >= 0 && y < maxY);
}

__global__ void
cudaComputeVelocity(bSimulator* sim) {

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= sim->dimX || y >= sim->dimY)
		return;

	unsigned long long int elementIdx = y * sim->dimX + x;

	bSimulator::node& n = *(sim->nodes + elementIdx);

	if (n.ntype == bSimulator::nodeType::BASE) {
		float macroVel[2];

		float density = sum(n.newDensities, 9);

		matMul(n.newDensities, sim->speeds, macroVel, 1, 9, 2);
		scalarProd(1.f / density, macroVel, macroVel, 2);
		scalarProd((float)sim->c, macroVel, macroVel, 2);
		n.vel = { macroVel[0], macroVel[1] };
	}


}

__global__ void
cudaComputeEquilibrium(bSimulator* sim) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= sim->dimX || y >= sim->dimY)
		return;

	unsigned long long int elementIdx = y * sim->dimX + x;

	bSimulator::node& n = *(sim->nodes + elementIdx);
	if (n.ntype == bSimulator::nodeType::BASE) {

		float density = sum(n.newDensities, 9);
		float macroVel[2] = { n.vel.x, n.vel.y };

		for (auto j = 0; j < 9; j++) {
			float dotProd = dot(&sim->speeds[2 * j], macroVel, 2);
			n.eqDensities[j] = density * sim->weights[j] * (1.f + 3.f * dotProd / sim->c
				+ 9.f * (pow(dotProd, 2) / sim->csqr) / 2.f
				- 3.f * dot(macroVel, macroVel, 2) / (2.f * sim->csqr));
		}
	}

}

__global__ void
cudaComputeNew(bSimulator* sim) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= sim->dimX || y >= sim->dimY)
		return;

	unsigned long long int elementIdx = y * sim->dimX + x;


	bSimulator::node& n = *(sim->nodes + elementIdx);
	if (n.ntype == bSimulator::nodeType::BASE) {
		float newDensities[9];

		vecSub(n.eqDensities, n.newDensities, newDensities, 9);
		scalarProd((float)sim->viscosity, newDensities, newDensities, 9);
		vecSum(newDensities, n.newDensities, newDensities, 9);
		memcpy(n.densities, newDensities, 9 * sizeof(float));
		memset(n.newDensities, 0.f, 9 * sizeof(float));
	}
}

__global__ void
cudaStream(bSimulator* sim) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= sim->dimX || y >= sim->dimY)
		return;

	unsigned long long int elementIdx = y * sim->dimX + x;
	bSimulator::node& n = *(sim->nodes + elementIdx);


	switch (n.ntype) {
	case bSimulator::nodeType::BASE: {
		for (int j = 0; j < 9; j++) {
			int dx = sim->directions[j][0];
			int dy = sim->directions[j][1];

			if (dx == 0 && dy == 0) {
				n.newDensities[j] = n.densities[j];
				continue;
			}

			long long int newX = n.x + dx;
			long long int newY = n.y + dy;

			if (!inside(x, y, sim->dimX, sim->dimY)) {
				switch (sim->doAtEdge) {

				case bSimulator::edgeBehaviour::LOOP: {
					newX = (newX + sim->dimX) % sim->dimX;
					newY = (newY + sim->dimY) % sim->dimY;

					bSimulator::node& nn = *(sim->nodes + newY * sim->dimX + newX);

					nn.newDensities[j] += n.densities[j];
					n.densities[j] = 0;
					break;
				}

				case bSimulator::edgeBehaviour::EXIT: {
					n.densities[j] = 0;
					break;
				}

				}

				continue;
			}

			bSimulator::node& nn = *(sim->nodes + newY * sim->dimX + newX);

			switch (nn.ntype) {
			case bSimulator::nodeType::BASE: {
				nn.newDensities[j] += n.densities[j];
				n.densities[j] = 0;
				break;
			}

			case bSimulator::nodeType::WALL: {
				int opposite = (j < 5) ? ((j - 1) + 2) % 4 + 1 : ((j - 5) + 2) % 4 + 5;
				n.newDensities[opposite] += n.densities[j];
				n.densities[j] = 0;
				break;
			}
			}

		}

		break;
	}

	case bSimulator::nodeType::WALL: {

		break;
	}
	}


}


extern "C" {
	void computeVelocity(bSimulator* sim) {
		cudaComputeVelocity <<< sim->gridDim, sim->blockDim >>> (sim);
		cudaDeviceSynchronize();
	}
	
	void computeEquilibrium(bSimulator* sim){
		cudaComputeEquilibrium <<< sim->gridDim, sim->blockDim >>> (sim);
		cudaDeviceSynchronize();
	}
	
	void computeNew(bSimulator* sim){
		cudaComputeNew <<< sim->gridDim, sim->blockDim >>> (sim);
		cudaDeviceSynchronize();
	}
	
	void stream(bSimulator* sim){
		cudaStream <<< sim->gridDim, sim->blockDim >>> (sim);
		cudaDeviceSynchronize();
	}
}
