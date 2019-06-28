#pragma once
#include <numeric>      

template <class type>
inline type mapNumber(type x, type inMin, type inMax, type outMin, type outMax) {
	return (type)( 
		(double)( (double) (x - inMin) * (double) (outMax - outMin) ) / (double)(inMax - inMin) + (double) outMin ) ;
}

template <class type>
inline void scalarProd(const type val, const type* inVec, type* outVec, const unsigned long int size) {
	for (auto i = 0; i < size; i++) {
		*(outVec + i) = *(inVec + i) * val;
	}
}

template <class type>
inline void vecSum(const type* aVec, const type* bVec, type* outVec, const unsigned long int size) {
	for (auto i = 0; i < size; i++) {
		*(outVec + i) = *(aVec + i) + *(bVec + i);
	}
}

template <class type>
inline void vecSub(const type* aVec, const type* bVec, type* outVec, const unsigned long int size) {
	for (auto i = 0; i < size; i++) {
		*(outVec + i) = *(aVec + i) - *(bVec + i);
	}
}

template <class type>
inline double magnitude(const type* inVec, const unsigned long int size) {
	return sqrt( (double)std::accumulate(inVec, inVec + size, 0.0, 
		[](const type& a, const type& b) { return a + pow(b, 2); } ) );
}

template <class type>
inline double average(const type* inVec, const unsigned long int size) {
	return (double)std::accumulate(inVec, inVec + size, 0.0) / (double)size;
}

template <class type>
inline type sum(const type* inVec, const unsigned long int size) {
	return std::accumulate(inVec, inVec + size, 0.0);
}

template <class type>
inline type dot(const type* aVec, const type* bVec, const unsigned long int size) {
	type prod = 0;
	for (auto i = 0; i < size; i++) {
		prod += *(aVec + i) * *(bVec + i);
	}

	return prod;
}

template <class type>
inline void matMul(const type* aMat, const type* bMat, type* outMat, const unsigned long int sizeM,
	const unsigned long int sizeN, const unsigned long int sizeP ) {
	
	for (auto i = 0; i < sizeM; i++) {
		for (auto j = 0; j < sizeP; j++) {
			type prod = 0;

			for (auto k = 0; k < sizeN; k++) {
				prod += *(aMat + i * sizeN + k) * *(bMat + k * sizeP + j);
			}

			*(outMat + i * sizeP + j) = prod;

		}
	}

}