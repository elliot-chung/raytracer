#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "device_launch_parameters.h"

#include "../Raytracer.hpp"
#include "../../linal.hpp"

#ifdef __INTELLISENSE__
template<class T>
void surf2Dwrite(T data, cudaSurfaceObject_t surfObj, int x, int y,
	cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);
#endif

#define BLOCK_SIZE 8
#define MAXIMUM_AA 4


struct GPURayHit
{
	bool didHit;
	unsigned int objectDataIndex;

	float distance;

	float4 hitPosition;
	float2 uv;
	mat4 tbnMatrix;

	GPUMaterial* material;
};

struct GPUMaterialPositionData
{
	float4 albedo;
	float4 normal;
	float3 ao;
	float roughness;
	float metal;
	float4 emission;
};

struct GPUTriangleHit
{
	float distance;
	float3 barycentricCoords;
};

struct GPURay
{
	float4	origin;
	float4	direction;
	int		bounceCount;
	float	maxDistance;
};

struct CameraParams
{
	float4	origin;
	float4	rotation;
	float   exposure;
	int		width;
	int 	height;

	cudaTextureObject_t rays;
};

struct SkyLightParams
{
	float4 direction;
	float4 lightColor;
	float4 skyColor;
};

struct RendererParams
{
	int bounceCount;
	float maxDistance;
	float aoIntensity;
	unsigned int frameCount;
	unsigned int progressiveFrameCount;
	bool antiAliasingEnabled;
	unsigned int sampleCount;
};


class GPURaytracer : public Raytracer
{
public:
	void raytrace(std::shared_ptr<Scene> s, std::shared_ptr<Camera> c, cudaSurfaceObject_t o);

}; 

// GPU kernel forward declarations
__global__ void raytraceKernel(CameraParams camera, cudaSurfaceObject_t canvas, const GPUObjectDataVector objectDataVector, const RendererParams renderer, const SkyLightParams skylight);

__device__ GPURay setupRay(const CameraParams& camera, const int x, const int y, const int bounceCount, const float maxDistance, const bool aaEnabled, unsigned int& seed);

__device__ float4 trace(GPURay& ray, const GPUObjectDataVector& objectDataVector, SkyLightParams skylight, const float aoIntensity, unsigned int& seed);

__device__ GPURayHit getIntersectionPoint(GPURay& ray, const GPUObjectDataVector& dataVector);

__device__ GPUMaterialPositionData getMaterialData(const GPURayHit& hit);

__device__ bool intersectsObjectBoundingBox(const GPURay& ray, const GPUObjectData& data); 

__device__ bool intersectsMeshBoundingBox(const GPURay& ray, const float3& minBounds, const float3& maxBounds, mat4 modelMatrix);

__device__ GPURayHit getIntersectionPoint(const GPURay& ray, const GPUObjectData& data);

__device__ GPUTriangleHit distToTriangle(const GPURay& ray, const float4& v0, const float4& v1, const float4& v2);

__device__ float4 fresnelSchlick(const float cosTheta, const float4& f0);

__device__ float distributionGGX(const float4& n, const float4& h, const float roughness); 

__device__ float geometrySchlickGGX(const float NdotV, const float roughness); 

__device__ float geometrySmith(const float4& n, const float4& v, const float4& l, const float roughness);

__device__ float4 getSkyLight(const float4& direction, const float4& lightDirection, const float4& lightColor, const float4& skyColor); 

__device__ __forceinline__ float4 exposureCorrection(const float4 color, const float exposure);

// GPU math functions

__device__ __forceinline__ float4 normalize(const float4 v);

__device__ __forceinline__ float3 matVecMul(const mat3 m, const float3 v);

__device__ __forceinline__ float4 matVecMul(const mat4 m, const float4 v);

__device__ __forceinline__ mat4 matMul(const mat4 a, const mat4 b);

__device__ __forceinline__ float4 cross(const float4 a, const float4 b);

__device__ __forceinline__ float dot(const float4 a, const float4 b);

__device__ __forceinline__ float4 negate(const float4 v);

__device__ __forceinline__ float4 lerp(const float4 a, const float4 b, const float t);

__device__ __forceinline__ float4 normalize(const float4 v);

__device__ __forceinline__ float3 normalize(const float3 v);

// Credit https://forums.developer.nvidia.com/t/cuda-for-quaternions-hyper-complex-numbers-operations/44116/2
__device__ __forceinline__ float4 rotate(const float4 v, const float4 q); 

__device__ __forceinline__ float randomValue(unsigned int& seed);

__device__ __forceinline__ float randomValueNormalDistribution(unsigned int& seed);

__device__ __forceinline__ float4 randomUnitVector(unsigned int& seed);

__device__ __forceinline__ float4 randomUnitVectorInHemisphere(unsigned int& seed, const float4& normal);

__device__ __forceinline__ float4 randomUnitVectorInCosineHemisphere(unsigned int& seed, const float4& normal);

__device__ __forceinline__ float4 reflect(const float4& v, const float4& normal);

__device__ __forceinline__ float smoothstep(const float edge0, const float edge1, float x);

__device__ __forceinline__ float clamp(const float x, const float a, const float b);





