#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "Mesh.hpp"



class DisplayObject
{
public:
	DisplayObject(glm::vec3 position = {}, glm::quat rotation = {}, glm::vec3 scale = {}) : minBound(0.0f), maxBound(0.0f)
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

	virtual void update() {}

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