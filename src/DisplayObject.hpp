#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "ImGui/imgui.h"

#include "Mesh.hpp"
#include "Material.hpp"

const glm::vec3 DEFAULT_POSITION = glm::vec3(0.0f);
const glm::quat DEFAULT_ROTATION = glm::quat();
const glm::vec3 DEFAULT_SCALE = glm::vec3(1.0f);

struct LLGPUObjectData // Long Lifespan GPU Object Data
{
	float3 minCompositeBounds; 
	float3 maxCompositeBounds; 

	GPUMeshData** meshes; 
	int* materialIndices; 
	GPUMaterial** materials; 
	int meshCount; 

	bool isComposite;
};


class DisplayObject
{
public:
	DisplayObject(glm::vec3 position = DEFAULT_POSITION, glm::quat rotation = DEFAULT_ROTATION, glm::vec3 scale = DEFAULT_SCALE)
	{
		Position = position;
		Rotation = rotation;
		Scale	 = scale;

		// reserve space for 1 mesh and 1 material
		materials = std::vector<Material*>(1);
		meshes = std::vector<std::pair<Mesh*, int>>(1);
	}

	~DisplayObject()
	{
		checkCudaErrors(cudaFree(gpuMeshes));
		checkCudaErrors(cudaFree(gpuMaterialIndices));
		checkCudaErrors(cudaFree(gpuMaterials));
		checkCudaErrors(cudaFree(gpuData));
	}

	glm::mat4 getModelMatrix()
	{
		glm::mat4 model = glm::mat4(1.0f);
		
		model = glm::translate(model, Position);
		model = glm::mat4_cast(Rotation) * model;
		model = glm::scale(model, Scale);

		return model;
	}

	virtual void update(ImGuiIO& io) {}
	virtual void updateGUI(ImGuiIO& io) {}

	inline void setPosition(glm::vec3 position)		{ Position = position; }
	inline void setRotation(glm::quat rotation)		{ Rotation = rotation; }
	inline void setScale(glm::vec3 scale)			{ Scale = scale; }

	inline Mesh* getMesh() { return meshes[0].first; }
	inline Material* getMaterial() { return materials[meshes[0].second]; }
	inline GPUMaterial* getGPUMaterial() { return materials[0]->getGPUMaterial(); }
	inline LLGPUObjectData* getGPUData() { return gpuData; }

	inline void setMaterialName(std::string name) { materials[0] = Material::getMaterial(name); } 

	inline bool isCompositeObject() { return isComposite; }
	inline float3 getMaxBound() { return compositeMaxBounds; } 
	inline float3 getMinBound() { return compositeMinBounds; }

	inline std::vector<std::pair<Mesh*, int>> getMeshes() { return meshes; }
	inline std::vector<Material*> getMaterials() { return materials; }

	void sendToGPU()
	{
		LLGPUObjectData* hostData = new LLGPUObjectData();

		hostData->minCompositeBounds = compositeMinBounds;
		hostData->maxCompositeBounds = compositeMaxBounds;

		hostData->meshCount = meshes.size();

		GPUMeshData** meshesHost = new GPUMeshData * [meshes.size()];
		int* materialIndicesHost = new int[meshes.size()];
		GPUMaterial** materialsHost = new GPUMaterial * [materials.size()];


		for (int i = 0; i < meshes.size(); i++)
		{
			meshesHost[i] = meshes[i].first->getGPUMeshData();
			materialIndicesHost[i] = meshes[i].second;
		}

		for (int i = 0; i < materials.size(); i++)
		{
			materialsHost[i] = materials[i]->getGPUMaterial();
		}



		checkCudaErrors(cudaMalloc((void**)&gpuMeshes, sizeof(GPUMeshData*) * meshes.size()));
		checkCudaErrors(cudaMalloc((void**)&gpuMaterialIndices, sizeof(int) * meshes.size()));
		checkCudaErrors(cudaMalloc((void**)&gpuMaterials, sizeof(GPUMaterial*) * materials.size()));

		checkCudaErrors(cudaMemcpy(gpuMeshes, meshesHost, sizeof(GPUMeshData*) * meshes.size(), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(gpuMaterialIndices, materialIndicesHost, sizeof(int) * meshes.size(), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(gpuMaterials, materialsHost, sizeof(GPUMaterial*) * materials.size(), cudaMemcpyHostToDevice));

		delete[] meshesHost;
		delete[] materialIndicesHost;
		delete[] materialsHost;

		hostData->isComposite = isComposite;

		hostData->meshes = gpuMeshes;
		hostData->materialIndices = gpuMaterialIndices;
		hostData->materials = gpuMaterials;

		checkCudaErrors(cudaMalloc((void**)&gpuData, sizeof(LLGPUObjectData)));
		checkCudaErrors(cudaMemcpy(gpuData, hostData, sizeof(LLGPUObjectData), cudaMemcpyHostToDevice));

		delete hostData;
	}
protected:

	glm::vec3 Position;
	glm::quat Rotation;
	glm::vec3 Scale;

	float3 compositeMaxBounds;
	float3 compositeMinBounds;

	std::vector<std::pair<Mesh*, int>> meshes; // Mesh, Material Index
	std::vector<Material*> materials;

	bool isComposite = false;

	LLGPUObjectData* gpuData;
	GPUMeshData** gpuMeshes; 
	int* gpuMaterialIndices; 
	GPUMaterial** gpuMaterials; 

	void updateCompositeBounds()
	{
		if (!isComposite)
		{
			compositeMaxBounds = meshes[0].first->getMaxBound(); 
			compositeMinBounds = meshes[0].first->getMinBound();  
			return;
		}

		compositeMaxBounds = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
		compositeMinBounds = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);

		for (auto meshPair : meshes)
		{
			Mesh* mesh = meshPair.first;

			float3 maxBounds = mesh->getMaxBound(); 
			float3 minBounds = mesh->getMinBound(); 

			compositeMaxBounds = make_float3(fmaxf(maxBounds.x, compositeMaxBounds.x), fmaxf(maxBounds.y, compositeMaxBounds.y), fmaxf(maxBounds.z, compositeMaxBounds.z));
			compositeMinBounds = make_float3(fminf(minBounds.x, compositeMinBounds.x), fminf(minBounds.y, compositeMinBounds.y), fminf(minBounds.z, compositeMinBounds.z));
		}
	}

	
};