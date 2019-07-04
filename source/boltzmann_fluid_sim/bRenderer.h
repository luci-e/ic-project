#pragma once
#include "bCommon.h"
#include "bSimulator.h"

class bRenderer : public Managed
{
	bSimulator* sim;

	enum renderMode {
		TEXTURE,
		POINTS
	} renderMode = renderMode::POINTS;


	bRenderer(bSimulator* sim) : sim(sim) {};
};

