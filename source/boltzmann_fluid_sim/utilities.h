#pragma once
#include <numeric>      

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif 


template <class type>
CUDA_CALLABLE inline type mapNumber(type x, type inMin, type inMax, type outMin, type outMax) {
	return (type)( 
		(double)( (double) (x - inMin) * (double) (outMax - outMin) ) / (double)(inMax - inMin) + (double) outMin ) ;
}

template <class type>
CUDA_CALLABLE inline void scalarProd(const type val, const type* inVec, type* outVec, const unsigned long int size) {
	for (unsigned long int i = 0; i < size; i++) {
		*(outVec + i) = *(inVec + i) * val;
	}
}

template <class type>
CUDA_CALLABLE inline void vecSum(const type* aVec, const type* bVec, type* outVec, const unsigned long int size) {
	for (unsigned long int i = 0; i < size; i++) {
		*(outVec + i) = *(aVec + i) + *(bVec + i);
	}
}

template <class type>
CUDA_CALLABLE inline void vecSub(const type* aVec, const type* bVec, type* outVec, const unsigned long int size) {
	for (unsigned long int i = 0; i < size; i++) {
		*(outVec + i) = *(aVec + i) - *(bVec + i);
	}
}

template <class type>
CUDA_CALLABLE inline double magnitude(const type* inVec, const unsigned long int size) {
	type sum = 0;
	for (unsigned long int i = 0; i < size; i++) {
		sum += pow( *(inVec + i), 2);
	}

	//return sqrt((double)std::accumulate(inVec, inVec + size, 0.0,
	//	[](const type& a, const type& b) { return a + pow(b, 2); }));

	return sqrt( sum );
}

template <class type>
CUDA_CALLABLE inline double average(const type* inVec, const unsigned long int size) {
	type sum = 0;
	for (unsigned long int i = 0; i < size; i++) {
		sum += *(inVec + i);
	}

	return (double)sum / (double)size;
	//return (double)std::accumulate(inVec, inVec + size, 0.0) / (double)size;
}

template <class type>
CUDA_CALLABLE inline type sum(const type* inVec, const unsigned long int size) {
	//return std::accumulate(inVec, inVec + size, 0.0);

	type sum = 0;
	for (unsigned long int i = 0; i < size; i++) {
		sum += *(inVec + i);
	}

	return sum;

}

template <class type>
CUDA_CALLABLE inline type dot(const type* aVec, const type* bVec, const unsigned long int size) {
	type prod = 0;
	for (unsigned long int i = 0; i < size; i++) {
		prod += *(aVec + i) * *(bVec + i);
	}

	return prod;
}

template <class type>
CUDA_CALLABLE inline void matMul(const type* aMat, const type* bMat, type* outMat, const unsigned long int sizeM,
	const unsigned long int sizeN, const unsigned long int sizeP ) {
	
	for (unsigned long int i = 0; i < sizeM; i++) {
		for (unsigned long int j = 0; j < sizeP; j++) {
			type prod = 0;

			for (unsigned long int k = 0; k < sizeN; k++) {
				prod += *(aMat + i * sizeN + k) * *(bMat + k * sizeP + j);
			}

			*(outMat + i * sizeP + j) = prod;

		}
	}

}