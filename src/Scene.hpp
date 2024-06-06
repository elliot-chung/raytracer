#pragma once

#include <unordered_map>
#include <glm/gtc/quaternion.hpp>
#include "DisplayObject.hpp"

#include "Cube.hpp"
//#include "Sphere.hpp"
//#include "CustomModel.hpp"

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
		ImGui::Separator();
		if (ImGui::Button("Add Object"))
			ImGui::OpenPopup("Add Object");
		if (ImGui::BeginPopup("Add Object"))
		{
			static int item = 0;
			static const char* items[] = { "Cube", "Sphere", "Custom Model" };
			ImGui::Combo("Type", &item, items, IM_ARRAYSIZE(items));

			static glm::vec3 position(0);
			static glm::quat rotation(1, 0, 0, 0); 
			static glm::vec3 euler(0); 
			static glm::vec3 scale(1);
			ImGui::DragFloat3("Position", &position[0], 0.1f);
			if (ImGui::DragFloat3("Rotation", &euler[0], 1.0f, -359.9f, 359.9f))
			{
				rotation = glm::quat(glm::radians(euler)); 
			}
			ImGui::DragFloat3("Scale", &scale[0], 0.1f); 
			


			static int cubeCount = 0;
			static int sphereCount = 0;
			static int customModelCount = 0;

			switch (item)
			{
			case 0:
				if (ImGui::Button("Add Cube"))
				{
					std::string name = "Cube " + std::to_string(cubeCount++);
					Cube* cube = new Cube(position, rotation, scale);
					cube->setMaterialName("Material");
					cube->sendToGPU(); 
					addToScene(name, cube); 
					ImGui::CloseCurrentPopup();
				}
				break;
			case 1:
				if (ImGui::Button("Add Sphere"))
				{
					//addToScene("Sphere", new Sphere());
					ImGui::CloseCurrentPopup();
				}
				break;
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