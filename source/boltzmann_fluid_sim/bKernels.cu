#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <string.h>

#include "utilities.h"
#include "bKernels.cuh"
#include "bCommon.h"
#include "bSimulator.h"
#include "bRenderer.h"

__device__ inline bool inside(long long int x, long long int y, unsigned long long int maxX, unsigned long long int maxY){
	return (x >= 0 && x < maxX && y >= 0 && y < maxY);
}

__global__ void
cudaComputeVelocity(bSimulator* sim) {

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= sim->dimX || y >= sim->dimY)
		return;

	unsigned long long int elementIdx = y * sim->dimX + x;

	node& n = *(sim->nodes + elementIdx);

	if (n.ntype == nodeType::BASE) {
		float macroVel[2];
		float density = sum(n.newDensities, 9);

		if (density > 0.f) {
			matMul(n.newDensities, sim->speeds, macroVel, 1, 9, 2);
			scalarProd((float) sim->c / density, macroVel, macroVel, 2);
			n.vel = { macroVel[0], macroVel[1] };
		}
		else {
			n.vel = {0.f, 0.f};
		}

	}
}

__global__ void
cudaComputeEquilibrium(bSimulator* sim) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= sim->dimX || y >= sim->dimY)
		return;

	unsigned long long int elementIdx = y * sim->dimX + x;

	node& n = *(sim->nodes + elementIdx);
	if (n.ntype == nodeType::BASE) {

		float density = sum(n.newDensities, 9);
		float macroVel[2] = { n.vel.x, n.vel.y };

		for (auto j = 0; j < 9; j++) {
			float dotProd = dot(&sim->speeds[2 * j], macroVel, 2);
			n.eqDensities[j] = density * sim->weights[j] * (1.f + 3.f * dotProd
				+ 9.f * (pow(dotProd, 2)) / 2.f
				- 3.f * dot(macroVel, macroVel, 2) / 2.f);
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


	node& n = *(sim->nodes + elementIdx);
	if (n.ntype == nodeType::BASE) {

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
	node& n = *(sim->nodes + elementIdx);


	switch (n.ntype) {
	case nodeType::BASE: {
		for (int j = 0; j < 9; j++) {
			int dx = sim->directions[j][0];
			int dy = sim->directions[j][1];

			if (dx == 0 && dy == 0) {
				n.newDensities[j] = n.densities[j];
				continue;
			}

			int opposite = (j < 5) ? ((j - 1) + 2) % 4 + 1 : ((j - 5) + 2) % 4 + 5;

			long long int newX = n.x + dx;
			long long int newY = n.y + dy;

			node* nn = nullptr;

			if (!inside(newX, newY, sim->dimX, sim->dimY)) {
				switch (sim->doAtEdge) {

				case bSimulator::edgeBehaviour::LOOP: {
					newX = (newX + sim->dimX) % sim->dimX;
					newY = (newY + sim->dimY) % sim->dimY;

					nn = (sim->nodes + newY * sim->dimX + newX);
					break;
				}

				case bSimulator::edgeBehaviour::EXIT: {
					n.newDensities[j] = 0.f;
					continue;
				}

				case bSimulator::edgeBehaviour::WALL: {
					goto wall;
				}

				}
			}
			else {
				nn = (sim->nodes + newY * sim->dimX + newX);
			}

			switch (nn->ntype) {
			case nodeType::BASE: {
				n.newDensities[opposite] += nn->densities[opposite];
				break;
			}

			case nodeType::WALL: {
				wall:
				n.newDensities[opposite] += n.densities[j];
				break;
			}

			case nodeType::SOURCE: {
				n.newDensities[opposite] += nn->densities[opposite];
				break;
			}

			case nodeType::SINK: {
				n.newDensities[j] = 0;
				break;
			}
			}

		}

		break;
	}

	case nodeType::WALL: {

		break;
	}
	}


}

__global__ void
cudaUpdateGraphics(bRenderer* simR)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= simR->sim->dimX || y >= simR->sim->dimY)
		return;

	unsigned long long int elementIdx = y * simR->sim->dimX + x;

	node& n = *(simR->sim->nodes + elementIdx);
	bRenderer::displayNode& dn = *(simR->cudaGLNodes + elementIdx);

	switch (n.ntype) {
	case nodeType::BASE: {
		float totalDensity = sum(&n.densities[0], 9);
		dn.density = 1.f;

		float newSpeeds[2] = { n.vel.x, n.vel.y };
		double mag = magnitude(newSpeeds, 2);

		if (mag > 0) {
			dn.vel.x = mapNumber<float>(newSpeeds[0] / mag, -1.f, 1.f, 0.f, 1.f);
			dn.vel.y = mapNumber<float>(newSpeeds[1] / mag, -1.f, 1.f, 0.f, 1.f);
		}
		else {
			dn.vel.x = mapNumber<float>(0.f, -1.f, 1.f, 0.f, 1.f);
			dn.vel.y = mapNumber<float>(0.f, -1.f, 1.f, 0.f, 1.f);
		}

		break;
	}

	case nodeType::WALL: {
		dn.density = 0.f;
		dn.vel = { 0.f, 0.f };
		break;
	}

	case nodeType::SOURCE: {
		dn.density = 1.f;
		dn.vel = { 1.f, 0.f };
		break;
	}

	case nodeType::SINK: {
		dn.density = 1.f;
		dn.vel = { 0.f, 1.f };
		break;
	}
	}

}


extern "C" {
	void computeVelocity(bSimulator* sim) {
		cudaComputeVelocity << < sim->gridDim, sim->blockDim >> > (sim);
	}
	
	void computeEquilibrium(bSimulator* sim){
		cudaComputeEquilibrium << < sim->gridDim, sim->blockDim >> > (sim);
	}
	
	void computeNew(bSimulator* sim){
		cudaComputeNew << < sim->gridDim, sim->blockDim >> > (sim);
	}
	
	void stream(bSimulator* sim){
		cudaStream <<< sim->gridDim, sim->blockDim >>> (sim);
	}

	void updateGraphics(bRenderer* simR)
	{
		cudaUpdateGraphics << < simR->sim->gridDim, simR->sim->blockDim >> > (simR);
	}
}
