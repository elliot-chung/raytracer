#pragma once

#include "DisplayObject.hpp"

class Cube : public DisplayObject
{
public:
	Cube(glm::vec3 position = DEFAULT_POSITION, glm::quat rotation = DEFAULT_ROTATION, glm::vec3 scale = DEFAULT_SCALE);
};