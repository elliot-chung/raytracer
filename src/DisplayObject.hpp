#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "ImGui/imgui.h"

#include "Mesh.hpp"
#include "Material.hpp"
#include "linal.hpp"

const glm::vec3 DEFAULT_POSITION = glm::vec3(0.0f);
const glm::quat DEFAULT_ROTATION = glm::quat();
const glm::vec3 DEFAULT_SCALE = glm::vec3(1.0f);

struct LLGPUObjectData // Long Lifespan GPU Object Data
{
	GPUMeshData** meshes; 
	int* materialIndices; 
	GPUMaterial** materials; 
	int meshCount; 

	bool isComposite;
};

struct GPUObjectData
{
	mat4 modelMatrix;

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

	inline void setMaterialName(std::string name) { materials[0] = Material::getMaterial(name); } 

	inline bool isCompositeObject() { return isComposite; }
	inline float3 getMaxBound() { return compositeMaxBounds; } 
	inline float3 getMinBound() { return compositeMinBounds; }

	inline std::vector<std::pair<Mesh*, int>> getMeshes() { return meshes; }
	inline std::vector<Material*> getMaterials() { return materials; }
protected:

	glm::vec3 Position;
	glm::quat Rotation;
	glm::vec3 Scale;

	float3 compositeMaxBounds;
	float3 compositeMinBounds;

	std::vector<std::pair<Mesh*, int>> meshes; // Mesh, Material Index
	std::vector<Material*> materials;

	bool isComposite = false;

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

			printf("Max Bounds: %f, %f, %f\n", maxBounds.x, maxBounds.y, maxBounds.z);
			printf("Min Bounds: %f, %f, %f\n", minBounds.x, minBounds.y, minBounds.z);

			compositeMaxBounds = make_float3(fmaxf(maxBounds.x, compositeMaxBounds.x), fmaxf(maxBounds.y, compositeMaxBounds.y), fmaxf(maxBounds.z, compositeMaxBounds.z));
			compositeMinBounds = make_float3(fminf(minBounds.x, compositeMinBounds.x), fminf(minBounds.y, compositeMinBounds.y), fminf(minBounds.z, compositeMinBounds.z));
			
		}
	}

	void sendToGPU()
	{
		
	}
};