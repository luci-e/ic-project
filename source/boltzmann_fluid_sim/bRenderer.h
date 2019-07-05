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

	struct textureNode {
		float2 pos;
		float2 texCoord;
	};

	enum renderMode {
		TEXTURE,
		POINTS
	} renderM = renderMode::TEXTURE;

	std::map < renderMode, Shader > shaders;

	struct cudaGraphicsResource* cudaVboNodes = NULL; // handles OpenGL-CUDA exchange
	displayNode* cudaGLNodes = NULL;
	size_t cudaGLNodesSize;

	GLuint nodesBuffer = 0;         // OpenGL array buffer object
	GLuint nodesVao = 0;            // OpenGL vertex buffer object

	float rectangleVertices[4 * (2+2)] = {
		// positions          // colors           // texture coords
		 0.5f,  0.5f,   1.0f, 1.0f, // top right
		 0.5f, -0.5f,  1.0f, 0.0f, // bottom right
		-0.5f, -0.5f,  0.0f, 0.0f, // bottom left
		-0.5f,  0.5f,    0.0f, 1.0f  // top left 
	};

	unsigned int rectangleIndices[6] = {
		0, 1, 3, // first triangle
		1, 2, 3  // second triangle
	};

	GLuint textureNodes = 0;		// OpenGL texture object

	GLuint textureEbo = 0;			// OpenGL element buffer object
	GLuint textureVbo = 0;			// OpenGL texture object
	GLuint textureVao = 0;			// OpenGL vertex buffer object

	bRenderer(bSimulator* sim) : sim(sim) {
		shaders[renderMode::POINTS] = Shader("pointsShader.glsl", "pointsFragment.glsl");
		shaders[renderMode::TEXTURE] = Shader("textureShader.glsl", "textureFragment.glsl");

		shaders[renderM].use();
	};

	void setRenderMode(renderMode mode);

	int initDisplayNodes();
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
