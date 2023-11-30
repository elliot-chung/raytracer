#pragma once

#include <vector>

#include "ImGui/imgui.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>



class Camera
{
public:
    Camera(glm::vec3& position, glm::quat& rotation, float fov) : Position(position), Rotation(rotation), verticalFOV(fov) { }

    // TODO: Move this to the ray tracing backend
    std::vector<glm::vec3> getRays(int width, int height)
    {
        static int sWidth = width;
        static int sHeight = height;
        static bool raysInitialized = false;

        if (sWidth != width || sHeight != height) raysInitialized = false;
        sWidth = width;
        sHeight = height;

        if (!raysInitialized)
        {
            raysInitialized = true;
            const float virtualHeight = glm::tan(glm::radians(verticalFOV / 2)) * 2;
            const float virtualWidth = virtualHeight * width / height;

            const float wStep = virtualWidth / width;
            const float hStep = virtualHeight / height;

            const float xOffset = wStep * (width / 2);
            const float yOffset = hStep * (height / 2);

            const glm::vec3 forward = glm::vec3(0.0f, 0.0f, -1.0f);
            const glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
            const glm::vec3 right = glm::vec3(1.0f, 0.0f, 0.0f);

            preTransformRays = std::vector<glm::vec3>(width * height, forward);
            for (int j = 0; j < height; j++) {
                for (int i = 0; i < width; i++) {
                    float rFactor = (float)((i * wStep) - xOffset);
                    float uFactor = (float)((j * hStep) - yOffset);

                    int index = j * width + i;
                    preTransformRays[index] = glm::normalize(rFactor * right + uFactor * up);
                }
            }
        }

        std::vector<glm::vec3> output(width * height, glm::vec3(0.0f));

        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                int index = j * width + i;
                output[index] = Rotation * preTransformRays[index];
            }
        }

        return output;
    }

    inline glm::vec3 getPosition() { return Position; }
    inline glm::quat getRotation() { return Rotation; }
    inline void setPosition(glm::vec3 position) { Position = position; }
    inline void setRotation(glm::quat rotation) { Rotation = rotation; }
private:
    glm::vec3 Position;
    glm::quat Rotation;

    std::vector<glm::vec3> preTransformRays;

    float verticalFOV;
};