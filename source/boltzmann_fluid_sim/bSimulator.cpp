#include "bSimulator.h"
#include <random>
#include <functional> 
#include <algorithm>
#include "utilities.h"
#include "bKernels.cuh"
#include <cassert>

void bSimulator::reset()
{
	std::default_random_engine generator;
	std::uniform_int_distribution<int> pos_distribution(1, 8);
	auto pos_dice = std::bind(pos_distribution, generator);

	// Initialize the nodes array
	for (auto y = 0; y < dimY; y++) {
		for (auto x = 0; x < dimX; x++) {
			node& n = *(nodes + (y * dimX + x));
			n.ntype = nodeType::BASE;
			n.x = x;
			n.y = y;
			memset(n.densities, 0.f, 9 * sizeof(float));
			memset(n.newDensities, 0.f, 9 * sizeof(float));
			memset(n.eqDensities, 0.f, 9 * sizeof(float));

			n.densities[0] = 1.f;

		}
	}

	CPUComputeVelocity();
	CPUcomputeEquilibrium();
}

void bSimulator::CPUUpdate()
{
	CPUstream();
	CPUComputeVelocity();
	CPUcomputeEquilibrium();
	CPUcomputeNew();
}

void bSimulator::CPUComputeVelocity()
{

	for (auto i = 0; i < totalPoints; i++) {
		node& n = *(nodes + i);

		if (n.ntype == nodeType::BASE) {
			float macroVel[2];

			float density = sum(n.newDensities, 9);

			matMul(n.newDensities, speeds, macroVel, 1, 9, 2);
			scalarProd(1.f / density, macroVel, macroVel, 2);
			scalarProd((float)c, macroVel, macroVel, 2);
			n.vel = { macroVel[0], macroVel[1] };
		}

	}

}

void bSimulator::CPUcomputeEquilibrium()
{
	for (auto i = 0; i < totalPoints; i++) {

		node& n = *(nodes + i);
		if (n.ntype == nodeType::BASE) {

			float density = sum(n.newDensities, 9);
			float macroVel[2] = { n.vel.x, n.vel.y };

			for (auto j = 0; j < 9; j++) {
				float dotProd = dot(&speeds[2 * j], macroVel, 2);
				n.eqDensities[j] = density * weights[j] * (1.f + 3.f * dotProd / c
					+ 9.f * (pow(dotProd, 2) / csqr) / 2.f
					- 3.f * dot(macroVel, macroVel, 2) / (2.f * csqr));
			}
		}

	}
}

void bSimulator::CPUcomputeNew()
{

	for (auto i = 0; i < totalPoints; i++) {
		node& n = *(nodes + i);
		if (n.ntype == nodeType::BASE) {
			float newDensities[9];

			vecSub(n.eqDensities, n.newDensities, newDensities, 9);
			scalarProd((float)viscosity, newDensities, newDensities, 9);
			vecSum(newDensities, n.newDensities, newDensities, 9);
			memcpy(n.densities, newDensities, 9 * sizeof(float));
			memset(n.newDensities, 0.f, 9 * sizeof(float));
		}
	}

}

void bSimulator::CPUstream()
{
	for (auto i = 0; i < totalPoints; i++) {
		node& n = *(nodes + i);

		switch (n.ntype) {
		case nodeType::BASE: {
			for (int j = 0; j < 9; j++) {
				int dx = directions[j][0];
				int dy = directions[j][1];

				if (dx == 0 && dy == 0) {
					n.newDensities[j] = n.densities[j];
					continue;
				}

				int opposite = (j < 5) ? ((j - 1) + 2) % 4 + 1 : ((j - 5) + 2) % 4 + 5;

				long long int newX = n.x + dx;
				long long int newY = n.y + dy;

				node *nn = nullptr;

				if (!inside(newX, newY)) {
					switch (doAtEdge) {

					case bSimulator::edgeBehaviour::LOOP: {
						newX = (newX + dimX) % dimX;
						newY = (newY + dimY) % dimY;

						nn = (nodes + newY * dimX + newX);
						break;
					}

					case bSimulator::edgeBehaviour::EXIT: {
						n.newDensities[j] = 0;
						continue;
					}

					}
				}
				else {
					nn = (nodes + newY * dimX + newX);
				}

				switch (nn->ntype) {
				case nodeType::BASE: {
					n.newDensities[opposite] += nn->densities[opposite];
					break;
				}

				case nodeType::WALL: {
					n.newDensities[opposite] += n.densities[j];
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

}

void bSimulator::GPUUpdate()
{
	cudaMemPrefetchAsync(nodes, totalPoints * sizeof(node), 0);

	GPUstream();
	GPUComputeVelocity();
	GPUcomputeEquilibrium();
	GPUcomputeNew();
}

void bSimulator::GPUComputeVelocity()
{
	computeVelocity(this);
	cudaDeviceSynchronize();
}

void bSimulator::GPUcomputeEquilibrium()
{
	computeEquilibrium(this);
	cudaDeviceSynchronize();
}

void bSimulator::GPUcomputeNew()
{
	computeNew(this);
	cudaDeviceSynchronize();
}

void bSimulator::GPUstream()
{
	stream(this);
	cudaDeviceSynchronize();
}

void bSimulator::setEdgeBehaviour(edgeBehaviour behaviour)
{
	cudaDeviceSynchronize();
	doAtEdge = behaviour;
}

int bSimulator::initNodes()
{
	cudaMallocManaged(&nodes, sizeof(node) * totalPoints);
	cudaDeviceSynchronize();

	if (nodes == NULL) {
		return -1;
	}

	reset();

	return 0;
}

bool bSimulator::inside(long long int x, long long int y)
{
	return (x >= 0 && x < dimX && y >= 0 && y < dimY);
}

void bSimulator::cleanup()
{
	cudaDeviceSynchronize();
	cudaFree(nodes);
	cudaDeviceSynchronize();
}
