#include "bSimulator.h"
#include <random>
#include <functional> 
#include <algorithm>
#include "utilities.h"
#include "bKernels.h"

void bSimulator::CPUUpdate()
{
	CPUstream();
	CPUComputeVelocity();
	CPUcomputeEquilibrium();
	CPUcomputeNew();
	updateDisplayNodes();
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

				long long int newX = n.x + dx;
				long long int newY = n.y + dy;

				if (!inside(newX, newY)) {

					switch (doAtEdge) {

					case edgeBehaviour::LOOP: {
						newX = (newX + dimX) % dimX;
						newY = (newY + dimY) % dimY;

						node& nn = *(nodes + newY * dimX + newX);

						nn.newDensities[j] += n.densities[j];
						n.densities[j] = 0;
						break;
					}

					case edgeBehaviour::EXIT: {
						n.densities[j] = 0;
						break;
					}

					}

					continue;
				}

				node& nn = *(nodes + newY * dimX + newX);

				switch (nn.ntype) {
				case nodeType::BASE: {
					nn.newDensities[j] += n.densities[j];
					n.densities[j] = 0;
					break;
				}

				case nodeType::WALL: {
					int opposite = (j < 5) ? ((j - 1) + 2) % 4 + 1 : ((j - 5) + 2) % 4 + 5;
					n.newDensities[opposite] += n.densities[j];
					n.densities[j] = 0;
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
	GPUstream();
	GPUComputeVelocity();
	GPUcomputeEquilibrium();
	GPUcomputeNew();
	updateDisplayNodes();
}

void bSimulator::GPUComputeVelocity()
{
	computeVelocity(this);
}

void bSimulator::GPUcomputeEquilibrium()
{
	computeEquilibrium(this);
}

void bSimulator::GPUcomputeNew()
{
	computeNew(this);
}

void bSimulator::GPUstream()
{
	stream(this);
}


int bSimulator::initNodes()
{
	cudaMallocManaged(&nodes, sizeof(node) * totalPoints);
	cudaDeviceSynchronize();

	if (nodes == NULL) {
		return -1;
	}

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

			// ********************************************* //

			if (x > 20 && x < 100 && y > 20 && y < 100) {
				n.densities[0] = .9f;
				n.densities[1] = .1f;
			}
			else {
				n.densities[0] = 1.f;
			}

			if (x > 100 && x < 200 && y > 100 && y < 200) {
				n.ntype = nodeType::WALL;
			}

			// ********************************************* //
		}
	}

	initDisplayNodes();

	return 0;
}

int bSimulator::initDisplayNodes()
{
	glGenBuffers(1, &nodesBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, nodesBuffer);
	glBufferData(GL_ARRAY_BUFFER, totalPoints * sizeof(displayNode), NULL, GL_DYNAMIC_DRAW);

	displayNode* displayNodes = (displayNode*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

	if (displayNodes == NULL) {
		printf("Error while mapping the buffer into client memory!\n");
		return -1;
	}
	else {
		printf("Successfully mapped buffer into client memory!\n");
	}

	// Initialize the nodes that will be displayed
	for (auto y = 0; y < dimY; y++) {
		for (auto x = 0; x < dimX; x++) {
			displayNode& dn = *(displayNodes + y * dimX + x);
			node& n = *(nodes + y * dimX + x);
			initDisplayNode(n, dn);
		}
	}

	glUnmapBuffer(GL_ARRAY_BUFFER);

	GLint size = 0;
	glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &size);
	printf("Allocated buffer size %d\n", size);

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, nodesBuffer);

	// Set the positions
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(displayNode), NULL);
	glEnableVertexAttribArray(0);
	// Set the velocity
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(displayNode), (void*)(2 * sizeof(float)));
	glEnableVertexAttribArray(1);
	// Set the density
	glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(displayNode), (void*)(2 * sizeof(float) + 2 * sizeof(float)));
	glEnableVertexAttribArray(2);

	return 0;
}

int bSimulator::updateDisplayNodes()
{
	glBindBuffer(GL_ARRAY_BUFFER, nodesBuffer);
	displayNode* displayNodes = (displayNode*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

	if (displayNodes == NULL) {
		return -1;
	}

	// Initialize the nodes that will be displayed
	for (auto y = 0; y < dimY; y++) {
		for (auto x = 0; x < dimX; x++) {
			displayNode& dn = *(displayNodes + y * dimX + x);
			node& n = *(nodes + y * dimX + x);
			updateDisplayNode(n, dn);
		}
	}

	glUnmapBuffer(GL_ARRAY_BUFFER);
}

void bSimulator::initDisplayNode(const node& n, displayNode& dn)
{
	dn.ntype = n.ntype;

	dn.pos.x = mapNumber<float>(n.x, 0, dimX, -1.f, 1.f);
	dn.pos.y = mapNumber<float>(n.y, 0, dimY, -1.f, 1.f);

	updateDisplayNode(n, dn);
}

void bSimulator::updateDisplayNode(const node& n, displayNode& dn)
{

	switch (n.ntype) {
	case nodeType::BASE: {
		dn.density = mapNumber<float>(sum(&n.densities[0], 9), 0.f, 1.f, 0.f, 1.f);

		float newSpeeds[2] = { n.vel.x, n.vel.y };
		double mag = magnitude(newSpeeds, 2);

		dn.vel.x = mapNumber<float>(newSpeeds[0] / mag, -1.f, 1.f, 0.f, 1.f);
		dn.vel.y = mapNumber<float>(newSpeeds[1] / mag, -1.f, 1.f, 0.f, 1.f);
		break;
	}

	case nodeType::WALL: {
		dn.density = 1.f;
		dn.vel = { 0,0 };
		break;
	}
	}

}

bool bSimulator::inside(long long int x, long long int y)
{
	return (x >= 0 && x < dimX && y >= 0 && y < dimY);
}

void bSimulator::cleanup()
{
	cudaDeviceSynchronize();
	cudaFree(nodes);
}
