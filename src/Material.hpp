#pragma once

#include <glm/glm.hpp>

#include <unordered_map>
#include <string>
#include <cstring>

#include "ImGui/imgui.h"
#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "global.hpp"

class Texture;
typedef std::unordered_map<std::string, std::pair<int, Texture*>> TextureMap; // Format <path-name, <ref-count, texture-pointer>>

class Texture
{
public:
	Texture(char const* path);
	~Texture();

	glm::vec4 sampleTexture(float x, float y);

	inline int getWidth() { return width; }
	inline int getHeight() { return height; }
	inline int size() { return dataSize; }
	inline float* getData() { return data; }
	inline bool getLoadSuccess() { return loadSuccess; }
	inline cudaTextureObject_t* getGPUTexture() { return &gpuTexture; }
	inline std::string getPath() { return path; }

	static inline Texture* getTexture(char const* path) { return textureMap[path].second; }

private:
	std::string path = "";

	int width;
	int height;
	int nChannels;
	int dataSize;

	bool loadSuccess;

	float* data;

	cudaTextureObject_t gpuTexture = 0;

	static TextureMap textureMap;
};

// -------------------------------------------------------------------------------------------------------- //
struct GPUMaterial
{
	const float4 NORMAL = make_float4(0.0f, 0.0f, 1.0f, 0.0f);
	const float3 AO = make_float3(1.0f, 1.0f, 1.0f);

	float4 albedo;
	float3 roughness;
	float3 metal;
	float3 emissionColor;
	float emissionStrength;

	cudaTextureObject_t normalTexture = 0;
	cudaTextureObject_t albedoTexture = 0;
	cudaTextureObject_t roughnessTexture = 0;
	cudaTextureObject_t metalTexture = 0;
	cudaTextureObject_t ambientOcclusionTexture = 0;
	cudaTextureObject_t emissionTexture = 0;

	__device__  float4 getNormal(float x, float y);
	__device__  float4 getAlbedo(float x, float y);
	__device__  float getRoughness(float x, float y);
	__device__  float getMetal(float x, float y);
	__device__  float3 getAmbientOcclusion(float x, float y);
	__device__  float4 getEmission(float x, float y);
};

struct MaterialInputBuffers
{
	char albedoTexturePath[256];
	char normalTexturePath[256];
	char roughnessTexturePath[256];
	char metalTexturePath[256];
	char ambientOcclusionTexturePath[256];
	char emissionTexturePath[256];
};

class Material;
typedef std::unordered_map<std::string, Material*> MaterialMap;

class Material
{
public:
	Material(char const* name, glm::vec4 albedo = glm::vec4(0.9f, 0.9f, 0.9f, 1.0f), float roughness = 0.5f, float metal = 0.0f, glm::vec3 emissionColor = {}, float emission = 0);
	~Material();
	Material(const Material&) = delete;

	inline std::string getName() { return materialName; }
	inline bool getSuccess() { return success; }

	glm::vec3 getNormal(float x, float y);
	glm::vec4 getAlbedo(float x, float y);
	float getRoughness(float x, float y);
	float getMetal(float x, float y);
	glm::vec3 getAmbientOcclusion(float x, float y);
	glm::vec3 getEmissionColor(float x, float y);
	inline float getEmissionStrength() { return emissionStrength; }

	bool setNormalTexture(const char* path);
	bool setAlbedoTexture(const char* path);
	bool setRoughnessTexture(const char* path);
	bool setMetalTexture(const char* path);
	bool setAmbientOcclusionTexture(const char* path);
	bool setEmissionTexture(const char* path);

	void sendToGPU();

	inline static Material* getMaterial(std::string& name) { return materialMap[name]; }
	inline static GPUMaterial* getGPUMaterial(std::string& name) { return materialMap[name]->gpuMaterial; }
	inline static std::vector<Material*> getAllMaterials() { std::vector<Material*> materials; for (auto& m : materialMap) materials.push_back(m.second); return materials; }

	inline GPUMaterial* getGPUMaterial() { return gpuMaterial; }

	static void displayMaterialGUI(ImGuiIO& io)
	{
		ImGui::Begin("Materials");

		{ // Material Editor
			// Material Selector List
			static Material* selectedMaterial = 0;
			static std::string selectedMaterialName = "";
			static std::string comboLabel = "##Material Selector";
			
			
			if (ImGui::BeginCombo(comboLabel.c_str(), selectedMaterialName.c_str(), ImGuiComboFlags_PopupAlignLeft)) 
			{
				for (auto& m : materialMap)
				{
					bool isSelected = selectedMaterialName == m.first;
					if (ImGui::Selectable(m.first.c_str(), isSelected))
					{
						selectedMaterial = m.second; 
						selectedMaterialName = m.first;
					}
				}
				ImGui::EndCombo();
			}

			ImGui::Separator();

			if (selectedMaterial)
			{
				// Material Options
				if (ImGui::ColorEdit3("Albedo", &selectedMaterial->simpleAlbedo.r)) selectedMaterial->sendToGPU();
				if (ImGui::DragFloat("Roughness", &selectedMaterial->simpleRoughness.r, 0.01f, 0.0f, 1.0f)) selectedMaterial->sendToGPU();
				if (ImGui::DragFloat("Metal", &selectedMaterial->simpleMetal.r, 0.01f, 0.0f, 1.0f)) selectedMaterial->sendToGPU();
				if (ImGui::ColorEdit3("Emission Color", &selectedMaterial->simpleEmissionColor.r)) selectedMaterial->sendToGPU();
				if (ImGui::DragFloat("Emission Strength", &selectedMaterial->emissionStrength, 0.01f, 0.0f, 1000000.0f)) selectedMaterial->sendToGPU();

				ImGui::Separator();

				char* albedoTexturePath = selectedMaterial->materialInputBuffers.albedoTexturePath;
				std::string albedolabel = "##Albedo Texture Path" + selectedMaterialName;
				ImGui::InputTextWithHint(albedolabel.c_str(), "Albedo Texture Path", albedoTexturePath, 256);
				ImGui::SameLine();
				if (ImGui::Button("Set Texture##Albedo") && (!selectedMaterial->albedoTexture || selectedMaterial->albedoTexture->getPath() != albedoTexturePath))
				{
					if (selectedMaterial->setAlbedoTexture(albedoTexturePath))
						selectedMaterial->sendToGPU();
					else
						strcpy(albedoTexturePath, (selectedMaterial->albedoTexture ? selectedMaterial->albedoTexture->getPath().c_str() : "")); 
				}

				char* normalTexturePath = selectedMaterial->materialInputBuffers.normalTexturePath;
				std::string normallabel = "##Normal Texture Path" + selectedMaterialName;
				ImGui::InputTextWithHint(normallabel.c_str(), "Normal Texture Path", normalTexturePath, 256);
				ImGui::SameLine();
				if (ImGui::Button("Set Texture##Normal") && (!selectedMaterial->normalTexture || selectedMaterial->normalTexture->getPath() != normalTexturePath))
				{
					if (selectedMaterial->setNormalTexture(normalTexturePath))
						selectedMaterial->sendToGPU();
					else
						strcpy(normalTexturePath, (selectedMaterial->normalTexture ? selectedMaterial->normalTexture->getPath().c_str() : ""));
				}

				char* roughnessTexturePath = selectedMaterial->materialInputBuffers.roughnessTexturePath;
				std::string roughlabel = "##Roughness Texture Path" + selectedMaterialName;
				ImGui::InputTextWithHint(roughlabel.c_str(), "Roughness Texture Path", roughnessTexturePath, 256);
				ImGui::SameLine();
				if (ImGui::Button("Set Texture##Roughness") && (!selectedMaterial->roughnessTexture || selectedMaterial->roughnessTexture->getPath() != roughnessTexturePath))
				{
					if (selectedMaterial->setRoughnessTexture(roughnessTexturePath))
						selectedMaterial->sendToGPU();
					else
						strcpy(roughnessTexturePath, (selectedMaterial->roughnessTexture ? selectedMaterial->roughnessTexture->getPath().c_str() : ""));
				}

				char* metalTexturePath = selectedMaterial->materialInputBuffers.metalTexturePath;
				std::string label = "##Metal Texture Path" + selectedMaterialName;
				ImGui::InputTextWithHint(label.c_str(), "Metal Texture Path", metalTexturePath, 256);
				ImGui::SameLine();
				if (ImGui::Button("Set Texture##Metal") && (!selectedMaterial->metalTexture || selectedMaterial->metalTexture->getPath() != metalTexturePath))
				{
					if (selectedMaterial->setMetalTexture(metalTexturePath))
						selectedMaterial->sendToGPU();
					else
						strcpy(metalTexturePath, (selectedMaterial->metalTexture ? selectedMaterial->metalTexture->getPath().c_str() : ""));
				}

				char* ambientOcclusionTexturePath = selectedMaterial->materialInputBuffers.ambientOcclusionTexturePath;
				label = "##Ambient Occlusion Texture Path" + selectedMaterialName;
				ImGui::InputTextWithHint(label.c_str(), "Ambient Occlusion Texture Path", ambientOcclusionTexturePath, 256);
				ImGui::SameLine();
				if (ImGui::Button("Set Texture##Ambient") && (!selectedMaterial->ambientOcclusionTexture || selectedMaterial->ambientOcclusionTexture->getPath() != ambientOcclusionTexturePath))
				{
					if (selectedMaterial->setAmbientOcclusionTexture(ambientOcclusionTexturePath))
						selectedMaterial->sendToGPU();
					else
						strcpy(ambientOcclusionTexturePath, (selectedMaterial->ambientOcclusionTexture ? selectedMaterial->ambientOcclusionTexture->getPath().c_str() : ""));
				}

				char* emissionTexturePath = selectedMaterial->materialInputBuffers.emissionTexturePath;
				label = "##Emission Texture Path" + selectedMaterialName;
				ImGui::InputTextWithHint(label.c_str(), "Emission Texture Path", emissionTexturePath, 256);
				ImGui::SameLine();
				if (ImGui::Button("Set Texture##Emission") && (!selectedMaterial->emissionTexture || selectedMaterial->emissionTexture->getPath() != emissionTexturePath))
				{
					if (selectedMaterial->setEmissionTexture(emissionTexturePath))
						selectedMaterial->sendToGPU();
					else
						strcpy(emissionTexturePath, (selectedMaterial->emissionTexture ? selectedMaterial->emissionTexture->getPath().c_str() : ""));
				}
			}
			
		}


		ImGui::Separator();
		{ // Add Material
			static char newMaterialName[256] = "";
			ImGui::InputTextWithHint("##New Material Name", "New Material Name", newMaterialName, IM_ARRAYSIZE(newMaterialName));
			ImGui::SameLine();
			static int success = 0;
			if (ImGui::Button("Add Material"))
			{
				if (strlen(newMaterialName) == 0)
					success = 2;
				else
				{
					Material* m = new Material(newMaterialName);
					success = (int)!m->getSuccess();
					if (success == 0)
					{
						newMaterialName[0] = '\0';
						m->sendToGPU();
					}
					else if (success == 1)
					{
						delete m;
					}
				}
			}

			switch (success)
			{
			case 1:
				ImGui::Text("Material name already exists!");
				break;
			case 2:
				ImGui::Text("Material name cannot be empty!");
				break;
			}
		}

		ImGui::End();
	}

protected:
	const glm::vec3 NORMAL = glm::vec3(0.0f, 0.0f, 1.0f);
	const glm::vec3 AO = glm::vec3(1.0f);

	std::string materialName;

	glm::vec4 simpleAlbedo;
	glm::vec3 simpleRoughness;
	glm::vec3 simpleMetal;
	glm::vec3 simpleEmissionColor;

	Texture* normalTexture = 0;
	Texture* albedoTexture = 0;
	Texture* roughnessTexture = 0;
	Texture* metalTexture = 0;
	Texture* ambientOcclusionTexture = 0;
	Texture* emissionTexture = 0;

	GPUMaterial* gpuMaterial = 0;

	float emissionStrength;

	bool success = true;

	MaterialInputBuffers materialInputBuffers = {};

	static MaterialMap materialMap;
	
};



