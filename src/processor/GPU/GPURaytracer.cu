#include "GPURaytracer.hpp"


__global__ void raytraceKernel(cudaSurfaceObject_t canvas, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	float4 color = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	surf2Dwrite(color, canvas, x * sizeof(float4), y);
}

void GPURaytracer::trace(std::shared_ptr<Scene> scene, std::shared_ptr<Camera> camera, cudaSurfaceObject_t canvas)
{
	dim3 blockSize(16, 16);
	dim3 gridSize((camera->getWidth() + blockSize.x - 1) / blockSize.x, (camera->getHeight() + blockSize.y - 1) / blockSize.y);

	raytraceKernel<<<gridSize, blockSize>>>(canvas, camera->getWidth(), camera->getHeight());
}