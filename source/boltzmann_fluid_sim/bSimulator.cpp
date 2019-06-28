#include "bSimulator.h"
#include <random>
#include <functional> 
#include <algorithm>
#include "utilities.h"

void bSimulator::InitParticles()
{
	glGenBuffers(1, &nodesBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, nodesBuffer);
	glBufferData(GL_ARRAY_BUFFER, totalPoints * sizeof(particle), NULL, GL_DYNAMIC_DRAW);

	particle* p = (particle*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

	if (p == NULL) {
		printf("Error while mapping the buffer into client memory!\n");
	}
	else {
		printf("Successfully mapped buffer into client memory!\n");
	}

	std::default_random_engine generator;
	std::uniform_real_distribution<float> pos_distribution(-1.0, 1.0);
	std::uniform_real_distribution<float> col_distribution(0.0, 1.0);

	auto pos_dice = std::bind(pos_distribution, generator);
	auto col_dice = std::bind(col_distribution, generator);

	int i, j;

	for (i = 0; i < this->dimY; i++)
	{
		for (j = 0; j < this->dimX; j++)
		{
			particle pa = { pos_dice(), pos_dice(),  col_dice(),  col_dice() };
			*(p + (i * this->dimX + j)) = pa;
		}
	}

	glUnmapBuffer(GL_ARRAY_BUFFER);

	GLint size = 0;
	glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &size);
	printf("Allocated buffer size %d\n", size);

	glGenVertexArrays(1, &this->vao);
	glBindVertexArray(this->vao);
	glBindBuffer(GL_ARRAY_BUFFER, this->nodesBuffer);

	// Set the positions and stride 
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(particle), NULL);
	glEnableVertexAttribArray(0);
	// Set the colours and stride
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(particle), (void*)(2 * sizeof(float)));
	glEnableVertexAttribArray(1);

}

void bSimulator::CPUUpdate()
{
	CPUcomputeEquilibrium();
	CPUcomputeNew();
	CPUstream();
	updateDisplayNodes();
}

void bSimulator::CPUcomputeEquilibrium()
{
	for (auto i = 0; i < totalPoints; i++) {
		node& n = *(nodes + i);
		float density = sum(n.densities, 9);
		float macroVel[2];
		matMul(n.densities, speeds, macroVel, 1, 9, 2);

		if (n.vel.x != 0.f && n.vel.y != 0.f) {
			float addVel[2] = { n.vel.x, n.vel.y };
			n.vel = { 0, 0 };
			vecSum(macroVel, addVel, macroVel, 2);
		}

		for (auto j = 0; j < 9; j++) {
			float dotProd = dot(&speeds[2 * j], macroVel, 2);
			n.eqDensities[j] = density * weights[j] * (1.f + 3.f * dotProd / c
				+ 9.f * (pow(dotProd, 2) / csqr) / 2.f
				- 3.f * pow(magnitude(macroVel, 2), 2) / (2.f * csqr));
		}

	}
}

void bSimulator::CPUcomputeNew()
{

	for (auto i = 0; i < totalPoints; i++) {
		node& n = *(nodes + i);
		float newDensities[9];

		vecSub(n.eqDensities, n.densities, newDensities, 9);
		scalarProd((float)viscosity, newDensities, newDensities, 9);
		vecSum(newDensities, n.densities, newDensities, 9);
		memcpy(n.densities, newDensities, 9 * sizeof(float));

	}

}

void bSimulator::CPUstream()
{
	for (auto i = 0; i < totalPoints; i++) {
		node& n = *(nodes + i);

		for (int j = 0; j < 9; j++) {
			int dx = directions[j][0];
			int dy = directions[j][1];

			if (dx == 0 && dy == 0)
				continue;

			long long int newX = n.x + dx;
			long long int newY = n.y + dy;

			if (!inside(newX, newY)) {
				int opposite = (j < 5) ? (j + 2) % 4 : (j + 2) % 4 + 4;
				n.densities[opposite] += n.densities[j];
				n.densities[j] = 0;
				continue;
			}

			node& nn = *(nodes + newY * dimX + newX);

			nn.densities[j] += n.densities[j];
			n.densities[j] = 0;
		}
	}

}

int bSimulator::initNodes(float density)
{

	std::default_random_engine generator;
	std::uniform_real_distribution<float> pos_distribution(-2.0, 2.0);
	std::uniform_real_distribution<float> col_distribution(0.0, 1.0);

	auto pos_dice = std::bind(pos_distribution, generator);
	auto col_dice = std::bind(col_distribution, generator);


	nodes = (node*)malloc(sizeof(node) * totalPoints);

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
			scalarProd<float>(density, &weights[0], &n.densities[0], 9);
			n.vel = { mapNumber<float>(x, 0, dimX, 0.f, -300.f), 0.f};
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
	dn.density = mapNumber<float>(sum(&n.densities[0], 9), 0.f, 100.f, 0.f, 1.f);

	float newSpeeds[2];
	matMul(n.densities, speeds, newSpeeds, 1, 9, 2);
	double mag = magnitude(newSpeeds, 2);

	dn.vel.x = mapNumber<float>(newSpeeds[0] / mag, -1.f, 1.f, 0.f, 1.f);
	dn.vel.y = mapNumber<float>(newSpeeds[1] / mag, -1.f, 1.f, 0.f, 1.f);
}

inline bool bSimulator::inside(long long int x, long long int y)
{
	return (x >= 0 && x < dimX && y >= 0 && y < dimY);
}

