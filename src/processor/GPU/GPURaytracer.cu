#include "GPURaytracer.hpp"

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

__global__ void raytraceKernel(CameraParams camera, cudaSurfaceObject_t canvas)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= camera.width || y >= camera.height)
		return;

	float4 ray = tex2D<float4>(camera.rays, x, y);
	
	float4 color = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
	surf2Dwrite(ray, canvas, x * sizeof(float4), y);
}

void GPURaytracer::trace(std::shared_ptr<Scene> scene, std::shared_ptr<Camera> camera, cudaSurfaceObject_t canvas)
{
	CameraParams params = {};
	glm::vec3 position = camera->getPosition();
	glm::quat transform = camera->getRotation();
	float exposure = camera->getExposure();
	int width = camera->getWidth();
	int height = camera->getHeight();
	cudaTextureObject_t rays = camera->getCudaRays();

	vec3transfer(params.origin, position);
	vec4transfer(params.rotation, transform);
	params.exposure = exposure;
	params.width = width;
	params.height = height;
	params.rays = rays;

	dim3 blockSize(16, 16);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

	raytraceKernel<<<gridSize, blockSize>>>(params, canvas);
}