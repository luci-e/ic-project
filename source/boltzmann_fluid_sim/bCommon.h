#pragma once

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
