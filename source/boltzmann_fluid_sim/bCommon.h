
#ifndef __B_COMMON__
#define __B_COMMON__

#define SQRT2 1.41421356237

class Managed {
public:
	void* operator new(size_t len) {
		void* ptr;
		cudaMallocManaged(&ptr, len);
		cudaDeviceSynchronize();
		return ptr;
	}

	void operator delete(void* ptr) {
		cudaDeviceSynchronize();
		cudaFree(ptr);
	}
};

enum nodeType {
	BASE,
	WALL
};

struct node {
	nodeType ntype;
	long long int x, y;
	float densities[9];
	float newDensities[9];
	float eqDensities[9];

	float2 vel;
	float2 addVel;
};



#endif // !__B_COMMON_

