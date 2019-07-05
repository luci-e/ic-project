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
	shaders[renderM].use();
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

	// Set the positions
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(displayNode), NULL);
	glEnableVertexAttribArray(0);
	// Set the velocity
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(displayNode), (void*)(2 * sizeof(float)));
	glEnableVertexAttribArray(1);
	// Set the density
	glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(displayNode), (void*)(2 * sizeof(float) + 2 * sizeof(float)));
	glEnableVertexAttribArray(2);

	// ---------------------------------------------------------------------------------------------------- //
	glGenVertexArrays(1, &textureVao);
	glBindVertexArray(textureVao);

	// Set the positions
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), NULL);
	glEnableVertexAttribArray(0);
	// Set the texcoord
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
	glEnableVertexAttribArray(1);

	// Initialize objects for texture rendering 
	glGenBuffers(1, &textureVbo);
	glBindBuffer(GL_ARRAY_BUFFER, textureVbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(rectangleVertices), rectangleVertices, GL_DYNAMIC_DRAW);

	glGenBuffers(1, &textureEbo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, textureEbo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(rectangleIndices), rectangleIndices, GL_DYNAMIC_DRAW);
	
	glGenTextures(1, &textureNodes);
	glBindTexture(GL_TEXTURE_2D, textureNodes);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	float* sampleTex = (float*) malloc(4 * sim->dimX * sim->dimY * sizeof(float));
	for (auto i = 0; i < sim->totalPoints*4; i+=4) {
		*(sampleTex + i + 0) = 0.f;
		*(sampleTex + i + 1) = 1.f;
		*(sampleTex + i + 2) = 0.f;
		*(sampleTex + i + 3) = 1.f;
	}
	   
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, sim->dimX, sim->dimY, 0, GL_RGBA, GL_FLOAT, sampleTex);
	glGenerateMipmap(GL_TEXTURE_2D);

	free(sampleTex);
	return 0;
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
	dn.pos.x = mapNumber<float>(n.x, 0, sim->dimX, -1.f, 1.f);
	dn.pos.y = mapNumber<float>(n.y, 0, sim->dimY, -1.f, 1.f);

	updateDisplayNode(n, dn);
}

void bRenderer::updateDisplayNode(const node& n, displayNode& dn)
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
	glBindBuffer(GL_ARRAY_BUFFER, nodesBuffer);
	cudaGraphicsGLRegisterBuffer(&cudaVboNodes, nodesBuffer, cudaGraphicsMapFlagsWriteDiscard);
	auto err = cudaGetLastError();

	if( err != cudaSuccess){
		printf("Error while initializing cuda GL interop\n");
		return -1;
	}

	return 0;
}

void bRenderer::render()
{
	switch (renderM) {
	case renderMode::TEXTURE: {
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, textureNodes);
		glBindVertexArray(textureVao);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
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
	cudaGraphicsUnregisterResource(cudaVboNodes);
	glDeleteVertexArrays(1, &nodesVao);
	glDeleteBuffers(1, &nodesBuffer);
}
