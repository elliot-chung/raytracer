#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "ImGui/imgui.h"

#include "Mesh.hpp"

const glm::vec3 DEFAULT_POSITION = glm::vec3(0.0f);
const glm::quat DEFAULT_ROTATION = glm::quat();
const glm::vec3 DEFAULT_SCALE = glm::vec3(1.0f);

class DisplayObject
{
public:
	DisplayObject(glm::vec3 position = DEFAULT_POSITION, glm::quat rotation = DEFAULT_ROTATION, glm::vec3 scale = DEFAULT_SCALE) : minBound(0.0f), maxBound(0.0f)
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

	inline void setPosition(glm::vec3 position)		{ Position = position; }
	inline void setRotation(glm::quat rotation)		{ Rotation = rotation; }
	inline void setScale(glm::vec3 scale)			{ Scale = scale; }
	inline void setMaterialName(std::string name)	{ materialName = name; }

	inline glm::vec3 getMinBound() { return minBound; }
	inline glm::vec3 getMaxBound() { return maxBound; }

	inline Mesh* getMesh() { return mesh; }
	inline std::string getMaterialName() { return materialName; }
	inline void setMaterialName(std::string& name) { materialName = name; } 
protected:

	glm::vec3 Position;
	glm::quat Rotation;
	glm::vec3 Scale;

	glm::vec3 minBound;	// In world coordinates
	glm::vec3 maxBound;

	Mesh* mesh = 0;
	std::string materialName = "";
};