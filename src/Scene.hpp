#pragma once

#include <unordered_map>
#include "DisplayObject.hpp"
#include "linal.hpp"

struct GPUObjectData
{
	mat4 modelMatrix;
	mat4 inverseModelMatrix;

	LLGPUObjectData* llData;
};

struct GPUObjectDataVector
{
	GPUObjectData* data;
	int size;
	bool isCopy = false;

	GPUObjectDataVector() : data(nullptr), size(0) {}

	GPUObjectDataVector(const GPUObjectDataVector& other) {
		size = other.size;
		data = other.data;
		isCopy = true;
	}

	~GPUObjectDataVector() { 
		if (!isCopy)
			checkCudaErrors(cudaFree(data));
	}
};

class Scene
{
public:
	typedef std::unordered_map<std::string, DisplayObject*> ObjectMap;
	bool addToScene(std::string name, DisplayObject* object)
	{
		if (objectMap.find(name) == objectMap.end())
		{
			objectMap[name] = object;
			return true;
		}
		return false;
	}

	bool removeFromSceen(const char* name, DisplayObject* object)
	{
		return (bool) objectMap.erase(name);
	}

	
	void update(ImGuiIO& io)
	{
		for (auto& object : objectMap)
		{
			object.second->update(io);
		}
	}

	void updateGUI(ImGuiIO& io)
	{
		ImGui::Begin("Scene");
		for (auto& object : objectMap)
		{
			// Selectable list of objects
			if (ImGui::Selectable(object.first.c_str()))
			{
				object.second->toggleSelect();
			}
		}
		ImGui::End();
	}
	
	inline ObjectMap& getObjects() { return objectMap; }

	GPUObjectDataVector getGPUObjectDataVector();
	
	void sendToGPU() 
	{
		for (auto& object : objectMap)
		{
			object.second->sendToGPU();
		}
	}
private:
	ObjectMap objectMap = {};
};