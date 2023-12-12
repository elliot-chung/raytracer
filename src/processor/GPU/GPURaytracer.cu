#include "GPURaytracer.hpp"


__global__ void raytraceKernel(cudaSurfaceObject_t canvas, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	
	for (x = 0; x < width; x++)
	{
		for (y = 0; y < height; y++)
		{
			float4 color = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
			surf2Dwrite(color, canvas, x * sizeof(float4), y);
		}
	}
	
}

void GPURaytracer::trace(std::shared_ptr<Scene> scene, std::shared_ptr<Camera> camera, cudaSurfaceObject_t canvas)
{
	dim3 blockSize(1, 1);
	dim3 gridSize(1, 1);

	raytraceKernel<<<gridSize, blockSize>>>(canvas, camera->getWidth(), camera->getHeight());
}