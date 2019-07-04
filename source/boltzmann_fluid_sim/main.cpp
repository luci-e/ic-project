#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "bApp.h"

int main(int argc, char** argv)
{
	bApp app;

	if (int r = app.initGL() < 0) {
		return r;
	}

	app.initSim(argc, argv);
	app.start();
	app.cleanup();

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}