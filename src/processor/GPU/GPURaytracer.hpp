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

struct mat4
{
	float4 r0;
	float4 r1;
	float4 r2;
	float4 r3;
};

struct CameraParams
{
	float3	origin;
	float4	rotation;
	float   exposure;
	int		width;
	int 	height;

	cudaTextureObject_t rays;
};

#define vec3transfer(a, b) \
	a.x = b.x; \
	a.y = b.y; \
	a.z = b.z;

#define vec4transfer(a, b) \
	a.x = b.x; \
	a.y = b.y; \
	a.z = b.z; \
	a.w = b.w;

class GPURaytracer : public Raytracer
{
public:
	void raytrace(std::shared_ptr<Scene> s, std::shared_ptr<Camera> c, cudaSurfaceObject_t o);

private:
}; 

// GPU kernel forward declarations
__global__ void raytraceKernel(CameraParams camera, cudaSurfaceObject_t canvas);

// Credit https://forums.developer.nvidia.com/t/cuda-for-quaternions-hyper-complex-numbers-operations/44116/2
__device__ __forceinline__ float4 rotate(const float4 v, const float4 q);


