#include "bSimulator.h"
#include <random>
#include <functional> 

void bSimulator::InitParticles()
{


	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, totalPoints * sizeof(particle), NULL, GL_DYNAMIC_DRAW);

	particle* p = (particle*) glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

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
			particle pa = { pos_dice(), pos_dice(),  col_dice(),  col_dice(),  col_dice() };
			*(p + (i * this->dimX + j)) = pa;
		}


	}

	glUnmapBuffer(GL_ARRAY_BUFFER);

	GLint size = 0;
	glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &size);
	printf("Allocated buffer size %d\n", size);
	
	glGenVertexArrays(1, &this->vao);
	glBindVertexArray(this->vao);
	glBindBuffer(GL_ARRAY_BUFFER, this->vbo);
	
	// Set the positions and stride 
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(particle), NULL);
	glEnableVertexAttribArray(0);
	// Set the colours and stride
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(particle), (void*)( 2 * sizeof(float) ) );
	glEnableVertexAttribArray(1);
}
