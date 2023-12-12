#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "device_launch_parameters.h"

#include "../Raytracer.hpp"

#ifdef __INTELLISENSE__
template<class T>
void surf2Dwrite(T data, cudaSurfaceObject_t surfObj, int x, int y,
	cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);
#endif

class GPURaytracer : public Raytracer
{
public:
	void trace(std::shared_ptr<Scene> s, std::shared_ptr<Camera> c, cudaSurfaceObject_t o);

private:
}; 