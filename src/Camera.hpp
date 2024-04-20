#pragma once

#include <vector>
#include <iostream>
#include "ImGui/imgui.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "cuda_runtime.h"
#include "helper_cuda.h"

#include "global.hpp"


class Camera
{
public:
    Camera(glm::vec3& position, glm::quat& rotation, float fov, float exposure) : Position(position), Rotation(rotation), verticalFOV(fov), exposureValue(exposure) { Euler = glm::degrees(glm::eulerAngles(Rotation)); }

    
    virtual void update(ImGuiIO& io) 
    {
        if (preTransformRays.size() != width * height) {
			calcRays();
		}
    }

    void updateGUI(ImGuiIO& io)
    {
		ImGui::Begin("Camera");
        ImGui::DragFloat3("Position", &Position[0], 0.1f);
        if (ImGui::DragFloat3("Rotation", &Euler[0], 1.0f, -359.9f, 359.9f)) 
        {
            Rotation = glm::quat(glm::radians(Euler));
        }
        ImGui::DragFloat("Exposure", &exposureValue, 0.1f, 0.0f);
        ImGui::DragFloat("FOV", &verticalFOV, 1.0f, -179.9f, 179.9f);
        if (ImGui::Button("Recalculate Rays")) {
            calcRays();  
		}
		ImGui::End();
	}
    

    void calcRays()
    {
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
        if (usingGPU) {
            if (cudaPreTransformRays) checkCudaErrors(cudaDestroyTextureObject(cudaPreTransformRays));

            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
            cudaArray_t cuArray;
            checkCudaErrors(cudaMallocArray(&cuArray, &channelDesc, width, height));

            const size_t spitch = width * sizeof(glm::vec4);
            checkCudaErrors(cudaMemcpy2DToArray(cuArray, 0, 0, &preTransformRays[0].x, spitch, width * sizeof(glm::vec4), height, cudaMemcpyHostToDevice));
            

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
            checkCudaErrors(cudaCreateTextureObject(&cudaPreTransformRays, &resDesc, &texDesc, NULL));
            
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
    inline void setHeight(int height) { this->height = height; }
    inline void setWidth(int width) { this->width = width; }

    inline cudaTextureObject_t getCudaRays() { return cudaPreTransformRays; }
private:
    glm::vec3 Position;
    glm::quat Rotation;
    glm::vec3 Euler;

    std::vector<glm::vec4> preTransformRays;
    cudaTextureObject_t    cudaPreTransformRays = 0;

    int width = 0;
    int height = 0;

    float verticalFOV;
    float exposureValue;
};