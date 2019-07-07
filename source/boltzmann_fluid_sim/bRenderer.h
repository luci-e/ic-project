#ifndef __B_RENDERER__
#define __B_RENDERER__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
#include <map>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>

#include "shaderHelper.h"
#include "bCommon.h"
#include "bSimulator.h"


class bRenderer : public Managed
{
public:

	bSimulator* sim;

	struct displayNode {
		float2 pos;
		float2 vel;
		float density;
	};

	enum renderMode : int {
		MESH = 0,
		POINTS = 1
	} renderM = renderMode::MESH;

	Shader shader;

	struct cudaGraphicsResource* cudaVboNodes = NULL; // handles OpenGL-CUDA exchange
	displayNode* cudaGLNodes = NULL;
	size_t cudaGLNodesSize;

	GLuint nodesBuffer = 0;         // OpenGL array buffer object
	GLuint nodesVao = 0;            // OpenGL vertex buffer object
	GLuint nodesEbo = 0;			// OpenGL element buffer object

	unsigned long long totalTriangles = 0;
	unsigned int *triangleIndices;

	bRenderer(bSimulator* sim) : sim(sim) {
		shader = Shader("pointsShader.glsl", "pointsFragment.glsl");
		shader.use();
	};

	void setRenderMode(renderMode mode);

	int initDisplayNodes();
	void generateTriangleIndices();
	int updateDisplayNodes();
	void initDisplayNode(const node& n, displayNode& dn);
	void updateDisplayNode(const node& n, displayNode& dn);

	int CPUUpdateGraphics();
	int GPUUpdateGraphics();
	int initCudaOpenGLInterop();

	void render();
	void cleanup();
};


#endif // !__B_RENDERER__
