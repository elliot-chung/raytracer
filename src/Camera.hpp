#pragma once

#include <vector>

#include "ImGui/imgui.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>



class Camera
{
public:
    Camera(glm::vec3& position, glm::quat& rotation, float fov) : Position(position), Rotation(rotation), verticalFOV(fov) { }

    
    virtual void update(ImGuiIO& io) {}

    void updateGUI(ImGuiIO& io)
    {
        static glm::vec3 euler = glm::eulerAngles(Rotation);
		ImGui::Begin("Camera");
		ImGui::Text("Position: (%.2f, %.2f, %.2f)", Position.x, Position.y, Position.z);
		ImGui::Text("Rotation: (%.2f, %.2f, %.2f, %.2f)", Rotation.x, Rotation.y, Rotation.z, Rotation.w);
        ImGui::InputFloat3("Position", &Position[0]);
        ImGui::InputFloat3("Rotation", &euler[0]);
        if (euler != glm::eulerAngles(Rotation)) {
			Rotation = glm::quat(euler);
		}
		ImGui::End();
	}
    

    void calcRays(int width, int height)
    {
        this->width = width;
        this->height = height;

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
                preTransformRays[index] = glm::normalize(forward + rFactor * right + uFactor * up);
            }
        }
    }

    inline glm::vec3 getRayDirection(int x, int y) { return Rotation * preTransformRays[y * width + x]; } // Applying this rotation could be done on the GPU

    inline int getWidth() { return width; }
    inline int getHeight() { return height; }
    inline int getPixelCount() { return width * height; }

    inline glm::vec3 getPosition() { return Position; }
    inline glm::quat getRotation() { return Rotation; }
    inline void setPosition(glm::vec3 position) { Position = position; }
    inline void setRotation(glm::quat rotation) { Rotation = rotation; }
private:
    glm::vec3 Position;
    glm::quat Rotation;

    std::vector<glm::vec3> preTransformRays;
    int width = 0;
    int height = 0;

    float verticalFOV;
};