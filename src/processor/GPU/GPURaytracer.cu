#include "GPURaytracer.hpp"



void GPURaytracer::raytrace(std::shared_ptr<Scene> scene, std::shared_ptr<Camera> camera, cudaSurfaceObject_t canvas)
{
	CameraParams params = {};
	glm::vec3 position = camera->getPosition();
	glm::quat rotation = camera->getRotation();
	float exposure = camera->getExposure();
	int width = camera->getWidth();
	int height = camera->getHeight();
	cudaTextureObject_t rays = camera->getCudaRays();

	vec3transfer(params.origin, position);
	vec4transfer(params.rotation, rotation);
	params.exposure = exposure;
	params.width = width;
	params.height = height;
	params.rays = rays;

	dim3 blockSize(16, 16);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

	raytraceKernel<<<gridSize, blockSize>>>(params, canvas);

	cudaDeviceSynchronize();
}

__global__ void raytraceKernel(CameraParams camera, cudaSurfaceObject_t canvas)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= camera.width || y >= camera.height)
		return;


	float4 ray = tex2D<float4>(camera.rays, x, y);
	ray = rotate(ray, camera.rotation);

	surf2Dwrite(ray, canvas, x * sizeof(float4), y);
}

__device__ __forceinline__ float4 rotate(const float4 v, const float4 q)
{
	float t2 = q.w * q.x;
	float t3 = q.w * q.y;
	float t4 = q.w * q.z;
	float t5 = -q.x * q.x;
	float t6 = q.x * q.y;
	float t7 = q.x * q.z;
	float t8 = -q.y * q.y;
	float t9 = q.y * q.z;
	float t10 = -q.z * q.z;
	return make_float4(
		2.0f * ((t8 + t10) * v.x + (t6 - t4) * v.y + (t3 + t7) * v.z) + v.x,
		2.0f * ((t4 + t6) * v.x + (t5 + t10) * v.y + (t9 - t2) * v.z) + v.y,
		2.0f * ((t7 - t3) * v.x + (t2 + t9) * v.y + (t5 + t8) * v.z) + v.z, 
		v.w
	);
}