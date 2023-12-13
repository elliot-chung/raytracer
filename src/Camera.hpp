#pragma once

#include <vector>
#include <iostream>
#include "ImGui/imgui.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "cuda_runtime.h"

class Camera
{
public:
    Camera(glm::vec3& position, glm::quat& rotation, float fov, float exposure, bool cuda) : Position(position), Rotation(rotation), verticalFOV(fov), exposureValue(exposure), useGPU(cuda) { }

    
    virtual void update(ImGuiIO& io) {}

    void updateGUI(ImGuiIO& io)
    {
        static glm::vec3 euler = glm::degrees(glm::eulerAngles(Rotation));
		ImGui::Begin("Camera");
        ImGui::DragFloat3("Position", &Position[0], 0.1f);
        ImGui::DragFloat3("Rotation", &euler[0], 1.0f, -359.9f, 359.9f);
        ImGui::DragFloat("FOV", &verticalFOV, 1.0f, -179.9f, 179.9f);
        ImGui::DragFloat("Exposure", &exposureValue, 0.1f, 0.0f);
        if (euler != glm::degrees(glm::eulerAngles(Rotation))) {
			Rotation = glm::quat(glm::radians(euler));
		}
        if (ImGui::Button("Recalculate Rays")) {
            calcRays(width, height);
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

        const glm::vec4 forward = glm::vec4(0.0f, 0.0f, -1.0f, 0.0f);
        const glm::vec4 up = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
        const glm::vec4 right = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);

        preTransformRays = std::vector<glm::vec4>(width * height, forward);
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                float rFactor = (float)((i * wStep) - xOffset);
                float uFactor = (float)((j * hStep) - yOffset);

                int index = j * width + i;
                preTransformRays[index] = glm::normalize(forward + rFactor * right + uFactor * up);
            }
        }

        // If CUDA is enabled, send rays to GPU
        if (useGPU) {
            if (cudaPreTransformRays != 0) cudaDestroyTextureObject(cudaPreTransformRays);

            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
            cudaArray_t cuArray;
            cudaMallocArray(&cuArray, &channelDesc, width, height);

            const size_t spitch = width * sizeof(glm::vec4);
            cudaMemcpy2DToArray(cuArray, 0, 0, &preTransformRays[0].x, spitch, width * sizeof(glm::vec4), height, cudaMemcpyHostToDevice);
            

            // Specify texture
            struct cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(resDesc));
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = cuArray;

            // Specify texture object parameters
            struct cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(texDesc));
            texDesc.addressMode[0] = cudaAddressModeClamp;
            texDesc.addressMode[1] = cudaAddressModeClamp;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode = cudaReadModeElementType;
            texDesc.normalizedCoords = 0; 

            // Create texture object
            cudaCreateTextureObject(&cudaPreTransformRays, &resDesc, &texDesc, NULL);
        }
    }

    inline glm::vec3 getRayDirection(int x, int y) { return Rotation * preTransformRays[y * width + x]; } // Applying this rotation could be done on the GPU

    inline int getWidth() { return width; }
    inline int getHeight() { return height; }
    inline int getPixelCount() { return width * height; }
    inline float getExposure() { return exposureValue; }

    inline glm::vec3 getPosition() { return Position; }
    inline glm::quat getRotation() { return Rotation; }
    inline void setPosition(glm::vec3 position) { Position = position; }
    inline void setRotation(glm::quat rotation) { Rotation = rotation; }

    inline cudaTextureObject_t getCudaRays() { return cudaPreTransformRays; }
private:
    glm::vec3 Position;
    glm::quat Rotation;

    std::vector<glm::vec4> preTransformRays;
    cudaTextureObject_t    cudaPreTransformRays = 0;

    int width = 0;
    int height = 0;

    float verticalFOV;
    float exposureValue;

    bool useGPU = false;
};