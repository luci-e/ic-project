#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "shaderHelper.h"
#include "utilities.h"
#include "bRenderer.h"
#include "bSimulator.h"
#include "bKernels.cuh"

#include <random>
#include <functional> 
#include <algorithm>

#include <cassert>

void bRenderer::setRenderMode(renderMode mode)
{
	renderM = mode;
	cudaDeviceSynchronize();
}

int bRenderer::initDisplayNodes()
{
	// Initialize objects for point rendering 
	glGenBuffers(1, &nodesBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, nodesBuffer);
	glBufferData(GL_ARRAY_BUFFER, sim->totalPoints * sizeof(displayNode), NULL, GL_DYNAMIC_DRAW);

	displayNode* displayNodes = (displayNode*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);

	if (displayNodes == NULL) {
		printf("Error while mapping the buffer into client memory!\n");
		return -1;
	}
	else {
		printf("Successfully mapped buffer into client memory!\n");
	}

	// Initialize the nodes that will be displayed
	for (auto y = 0; y < sim->dimY; y++) {
		for (auto x = 0; x < sim->dimX; x++) {
			displayNode& dn = *(displayNodes + y * sim->dimX + x);
			node& n = *(sim->nodes + y * sim->dimX + x);
			initDisplayNode(n, dn);
		}
	}

	glUnmapBuffer(GL_ARRAY_BUFFER);

	GLint size = 0;
	glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &size);
	printf("Allocated buffer size %d\n", size);

	glGenVertexArrays(1, &nodesVao);
	glBindVertexArray(nodesVao);
	glBindBuffer(GL_ARRAY_BUFFER, nodesBuffer);

	glGenBuffers(1, &nodesEbo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, nodesEbo);
	generateTriangleIndices();
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3 * totalTriangles * sizeof(unsigned int), triangleIndices, GL_DYNAMIC_DRAW);

	// Set the positions
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(displayNode), NULL);
	glEnableVertexAttribArray(0);
	// Set the velocity
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(displayNode), (void*)(2 * sizeof(float)));
	glEnableVertexAttribArray(1);
	// Set the density
	glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(displayNode), (void*)(2 * sizeof(float) + 2 * sizeof(float)));
	glEnableVertexAttribArray(2);

}

void bRenderer::generateTriangleIndices()
{
	unsigned long long trianglesPerStrip = ((sim->dimX - 1) * 2);

	totalTriangles = ((sim->dimX - 1) * 2) * (sim->dimY - 1);

	triangleIndices = (unsigned int*)malloc(3 * totalTriangles * sizeof(unsigned int));

	for (auto i = 0; i < totalTriangles; i++) {

		unsigned int indices[3];
		unsigned int strip = i / trianglesPerStrip;
		unsigned long long normalizedIndex = i - (strip * trianglesPerStrip);

		if (normalizedIndex < trianglesPerStrip / 2) {
			indices[0] = normalizedIndex + strip * sim->dimX;
			indices[1] = normalizedIndex + 1 + strip * sim->dimX;
			indices[2] = normalizedIndex + (1 + strip) * sim->dimX;

			memcpy(triangleIndices + (i * 3), indices, 3 * sizeof(unsigned int));
		}
		else {

			indices[0] = normalizedIndex + (strip * sim->dimX) - (trianglesPerStrip / 2) + 1;
			indices[1] = normalizedIndex + (strip * sim->dimX) + 1;
			indices[2] = normalizedIndex + (strip * sim->dimX) + 2;

			memcpy(triangleIndices + (i * 3), indices, 3 * sizeof(unsigned int));
		}
	}
}

int bRenderer::updateDisplayNodes()
{
	glBindBuffer(GL_ARRAY_BUFFER, nodesBuffer);
	displayNode* displayNodes = (displayNode*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);

	if (displayNodes == NULL) {
		return -1;
	}

	// Initialize the nodes that will be displayed
	for (auto y = 0; y < sim->dimY; y++) {
		for (auto x = 0; x < sim->dimX; x++) {
			displayNode& dn = *(displayNodes + y * sim->dimX + x);
			node& n = *(sim->nodes + y * sim->dimX + x);
			updateDisplayNode(n, dn);
		}
	}

	glUnmapBuffer(GL_ARRAY_BUFFER);
}

void bRenderer::initDisplayNode(const node& n, displayNode& dn)
{
	dn.pos.x = mapNumber<float>(n.x, 0, sim->dimX-1, -1.f, 1.f);
	dn.pos.y = mapNumber<float>(n.y, 0, sim->dimY-1, 1.f, -1.f);

	updateDisplayNode(n, dn);
}

void bRenderer::updateDisplayNode(const node& n, displayNode& dn)
{
	switch (n.ntype) {
	case nodeType::BASE: {
		float totalDensity = sum(&n.densities[0], 9);
		dn.density = totalDensity / 3.f;

		float newSpeeds[2] = { n.vel.x, n.vel.y };
		double mag = magnitude(newSpeeds, 2);

		dn.vel.x = mapNumber<float>(newSpeeds[0] / mag, -1.f, 1.f, 0.f, 1.f);
		dn.vel.y = mapNumber<float>(newSpeeds[1] / mag, -1.f, 1.f, 0.f, 1.f);
		break;
	}

	case nodeType::WALL: {
		dn.density = 0.f;
		dn.vel = { 0,0 };
		break;
	}
	}
}

int bRenderer::CPUUpdateGraphics()
{
	updateDisplayNodes();
	return 0;
}

int bRenderer::GPUUpdateGraphics()
{
	cudaGraphicsMapResources(1, &cudaVboNodes);

	cudaGraphicsResourceGetMappedPointer((void**)& cudaGLNodes,
		&cudaGLNodesSize,
		cudaVboNodes);

	updateGraphics(this);
	cudaDeviceSynchronize();
	cudaGraphicsUnmapResources(1, &cudaVboNodes);
	return 0;
}

int bRenderer::initCudaOpenGLInterop()
{
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, nodesBuffer);
	cudaGraphicsGLRegisterBuffer(&cudaVboNodes, nodesBuffer, cudaGraphicsMapFlagsWriteDiscard);
	auto err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("Error while initializing cuda GL interop\n");
		return -1;
	}

	return 0;
}

void bRenderer::render()
{
	switch (renderM) {
	case renderMode::MESH: {
		glBindVertexArray(nodesVao);
		//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glDrawElements(GL_TRIANGLES, totalTriangles * 3, GL_UNSIGNED_INT, 0);
		break;
	}

	case renderMode::POINTS: {
		glBindVertexArray(nodesVao);
		glDrawArrays(GL_POINTS, 0, sim->totalPoints);
		break;
	}
	}

}

void bRenderer::cleanup()
{
	cudaDeviceSynchronize();
	cudaGraphicsUnregisterResource(cudaVboNodes);
	glDeleteVertexArrays(1, &nodesVao);
	glDeleteBuffers(1, &nodesBuffer);
	free(triangleIndices);
	cudaDeviceSynchronize();
}

