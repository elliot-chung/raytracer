#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "ImGui/imgui.h"

#include "Mesh.hpp"
#include "Material.hpp"

#include <algorithm>

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
	DisplayObject(glm::vec3 position = DEFAULT_POSITION, glm::quat rotation = DEFAULT_ROTATION, glm::vec3 scale = DEFAULT_SCALE);

	~DisplayObject();

	glm::mat4 getModelMatrix();

	inline glm::mat4 getInverseModelMatrix() { return glm::inverse(getModelMatrix()); } 

	virtual void update(ImGuiIO& io) {}
	virtual void updateGUI(ImGuiIO& io);

	inline void setPosition(glm::vec3 position)		{ Position = position; }
	inline void setRotation(glm::quat rotation)		{ Rotation = rotation; }
	inline void setScale(glm::vec3 scale)			{ Scale = scale; }

	inline Mesh* getMesh() { return meshes[0].first; }
	inline Material* getMaterial() { return materials[meshes[0].second]; }
	inline GPUMaterial* getGPUMaterial() { return materials[0]->getGPUMaterial(); }
	inline LLGPUObjectData* getGPUData() { return gpuData; }

	inline void setMaterialName(std::string name) { materials.push_back(Material::getMaterial(name)); } 
	inline void setMaterialNameVector(std::vector<std::string> names) { std::transform(names.begin(), names.end(), std::back_inserter(materials), Material::getMaterial); }

	inline bool isCompositeObject() { return isComposite; }
	inline float3 getMaxBound() { return compositeMaxBounds; } 
	inline float3 getMinBound() { return compositeMinBounds; }

	inline std::vector<std::pair<Mesh*, int>> getMeshes() { return meshes; }
	inline std::vector<Material*> getMaterials() { return materials; }

	inline void select() { selectedObject = this; }
	inline void deselect() { if (this == selectedObject) selectedObject = 0; }
	inline void toggleSelect() { if (this == selectedObject) selectedObject = 0; else selectedObject = this; } 
	
	inline bool getLoadSuccess() { return loadSuccess; }

	void sendToGPU();

	static void displaySelectedObjectGUI(ImGuiIO& io)
	{
		ImGui::Begin("Selected Object");
		if (selectedObject) selectedObject->updateGUI(io);
		else ImGui::Text("No object selected");
		ImGui::End();
	}
protected:

	glm::vec3 Position;
	glm::quat Rotation;
	glm::vec3 Euler;
	glm::vec3 Scale;

	float3 compositeMaxBounds;
	float3 compositeMinBounds;

	std::vector<std::pair<Mesh*, int>> meshes; // Mesh, Material Index
	std::vector<Material*> materials;

	bool isComposite = false;
	bool loadSuccess = false;

	static DisplayObject* selectedObject;

	LLGPUObjectData* gpuData = 0;
	GPUMeshData** gpuMeshes = 0;
	int* gpuMaterialIndices = 0;
	GPUMaterial** gpuMaterials = 0;

	void updateCompositeBounds();

	void copyHostLLData(const DisplayObject* other);

	
};