#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "ImGui/imgui.h"

#include "Mesh.hpp"
#include "Material.hpp"

const glm::vec3 DEFAULT_POSITION = glm::vec3(0.0f);
const glm::quat DEFAULT_ROTATION = glm::quat();
const glm::vec3 DEFAULT_SCALE = glm::vec3(1.0f);

class DisplayObject
{
public:
	DisplayObject(glm::vec3 position = DEFAULT_POSITION, glm::quat rotation = DEFAULT_ROTATION, glm::vec3 scale = DEFAULT_SCALE)
	{
		Position = position;
		Rotation = rotation;
		Scale	 = scale;
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

	inline Mesh* getMesh() { return mesh; }
	inline Material* getMaterial() { return material; }
	inline GPUMaterial* getGPUMaterial() { return material->getGPUMaterial(); }
	inline std::string getMaterialName() { return materialName; }

	inline void setMaterialName(std::string name) { materialName = name; material = Material::getMaterial(name); }
protected:

	glm::vec3 Position;
	glm::quat Rotation;
	glm::vec3 Scale;

	Mesh* mesh = 0;
	Material* material = 0;
	std::string materialName = "";
};