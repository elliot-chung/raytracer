#pragma once

#include <vector>
#include <string>

#include <glm/glm.hpp>

#include "cuda_runtime.h"
#include "helper_cuda.h"

#include "global.hpp"


struct GPUMeshData
{
	float* vertices;
	float* uvs;
	int* indices;
	float* normals;

	int triangleCount;

	float3 minBounds;
	float3 maxBounds;
};

class Mesh
{
public:
	Mesh(std::vector<float>& vertices, std::vector<int>& indices, std::vector<float>& uvCoords)
	{
		this->vertices		= vertices;
		this->indices	    = indices;
		this->uvCoords		= uvCoords;
		this->triangleCount = indices.size() / 3;
		calcBounds();

		if (availableGPU)
		{
			cudaVertices = uploadToGPU(cudaVertices, this->vertices);
			cudaUVCoords = uploadToGPU(cudaUVCoords, this->uvCoords);
			cudaIndices  = uploadToGPU(cudaIndices, this->indices);
			cudaNormals  = 0;

			GPUMeshData meshData;
			meshData.vertices = cudaVertices;
			meshData.uvs = cudaUVCoords;
			meshData.indices = cudaIndices;
			meshData.normals = cudaNormals;

			meshData.triangleCount = triangleCount;

			meshData.minBounds = make_float3(minBound.x, minBound.y, minBound.z);
			meshData.maxBounds = make_float3(maxBound.x, maxBound.y, maxBound.z);

			checkCudaErrors(cudaMalloc((void**)&gpuMeshData, sizeof(GPUMeshData)));
			checkCudaErrors(cudaMemcpy(gpuMeshData, &meshData, sizeof(GPUMeshData), cudaMemcpyHostToDevice));
		}
	}

	Mesh(std::vector<float>& vertices, std::vector<int>& indices, std::vector<float>& uvCoords, std::vector<float>& normals)
	{
		this->vertices = vertices; 
		this->indices = indices; 
		this->uvCoords = uvCoords; 
		this->normals = normals; 
		this->triangleCount = indices.size() / 3; 
		calcBounds(); 

		if (availableGPU)
		{
			cudaVertices = uploadToGPU(cudaVertices, this->vertices); 
			cudaUVCoords = uploadToGPU(cudaUVCoords, this->uvCoords); 
			cudaIndices = uploadToGPU(cudaIndices, this->indices); 
			cudaNormals = uploadToGPU(cudaNormals, this->normals); 

			GPUMeshData meshData; 
			meshData.vertices = cudaVertices; 
			meshData.uvs = cudaUVCoords; 
			meshData.indices = cudaIndices; 
			meshData.normals = cudaNormals; 

			meshData.triangleCount = triangleCount; 

			meshData.minBounds = make_float3(minBound.x, minBound.y, minBound.z); 
			meshData.maxBounds = make_float3(maxBound.x, maxBound.y, maxBound.z); 

			checkCudaErrors(cudaMalloc((void**)&gpuMeshData, sizeof(GPUMeshData))); 
			checkCudaErrors(cudaMemcpy(gpuMeshData, &meshData, sizeof(GPUMeshData), cudaMemcpyHostToDevice)); 
		} 
	}
	


	~Mesh()
	{
		if (availableGPU)
		{
			checkCudaErrors(cudaFree(cudaVertices));
			checkCudaErrors(cudaFree(cudaUVCoords));
			checkCudaErrors(cudaFree(cudaIndices));
			checkCudaErrors(cudaFree(gpuMeshData));
		}
	}

	inline float3 getMinBound() { return minBound; }
	inline float3 getMaxBound() { return maxBound; }

	inline std::vector<float> getVertices() { return vertices; }
	inline std::vector<float> getUVCoords() { return uvCoords; }
	inline std::vector<int>	  getIndices()  { return indices; }
	
	inline int getTriangleCount() { return triangleCount; }

	inline GPUMeshData* getGPUMeshData() { return gpuMeshData; }

private:
	std::vector<float>	vertices;
	std::vector<float>  uvCoords;
	std::vector<int>	indices;
	std::vector<float>  normals;

	float* cudaVertices = 0;
	float* cudaUVCoords = 0;
	int* cudaIndices = 0;
	float* cudaNormals = 0;

	GPUMeshData* gpuMeshData;

	float3 minBound = make_float3(FLT_MAX, FLT_MAX, FLT_MAX); // In local coordinates
	float3 maxBound = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

	int triangleCount = 0;

	void calcBounds()
	{
		for (int i = 0; i < triangleCount; i++)
		{
			float x = vertices[i];
			float y = vertices[i + 1];
			float z = vertices[i + 2];

			if (minBound.x > x) minBound.x = x;
			if (minBound.y > y) minBound.y = y;
			if (minBound.z > z) minBound.z = z;

			if (maxBound.x < x) maxBound.x = x;
			if (maxBound.y < y) maxBound.y = y;
			if (maxBound.z < z) maxBound.z = z;
		}
	}

	template<typename T>
	inline T* uploadToGPU(T* devPtr, std::vector<T>& data)
	{
		checkCudaErrors(cudaMalloc((void**)&devPtr, data.size() * sizeof(T)));
		checkCudaErrors(cudaMemcpy(devPtr, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice));
		return devPtr;
	}
};