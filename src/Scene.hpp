#pragma once

#include <unordered_map>
#include "DisplayObject.hpp"
#include "linal.hpp"

struct GPUObjectData
{
	mat4 modelMatrix;

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
		for (auto& object : objectMap)
		{
			object.second->updateGUI(io);
		}
	}
	
	inline ObjectMap& getObjects() { return objectMap; }

	GPUObjectDataVector getGPUObjectDataVector()
	{
		GPUObjectData* objectDataArray = new GPUObjectData[objectMap.size()];
		int i = 0;
		for (auto& objPair : objectMap)
		{
			DisplayObject* obj = objPair.second; 
			GPUObjectData data = {}; 

			mat4transfer(data.modelMatrix, obj->getModelMatrix()); 
			data.llData = obj->getGPUData(); 

			objectDataArray[i++] = data; 
		}
		GPUObjectData* objectDataArrayDev; 
		checkCudaErrors(cudaMalloc((void**)&objectDataArrayDev, sizeof(GPUObjectData) * objectMap.size())); 
		checkCudaErrors(cudaMemcpy(objectDataArrayDev, objectDataArray, sizeof(GPUObjectData) * objectMap.size(), cudaMemcpyHostToDevice)); 
		delete[] objectDataArray; 

		GPUObjectDataVector output; 
		output.data = objectDataArrayDev; 
		output.size = objectMap.size();

		return output;
	}
	
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