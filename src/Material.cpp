#include "Material.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

Material::Material(char const* name, glm::vec4 albedo, float roughness,
	float metal, glm::vec3 emissionColor, float emission)
{
	materialName = std::string(name);
	if (materialMap.find(materialName) != materialMap.end()) // Not unique
	{
		success = false;
		return;
	}
	
	simpleAlbedo = albedo;
	simpleRoughness = glm::vec3(roughness);
	simpleMetal = glm::vec3(metal);
	simpleEmissionColor = emissionColor;
	emissionStrength = emission;

	success = true;
	materialMap[materialName] = this;
}

Material::~Material()
{
	delete normalTexture;
	delete albedoTexture;
	delete roughnessTexture;
	delete metalTexture;
	delete ambientOcclusionTexture;
	delete emissionTexture;

	materialMap.erase(materialName);
}

glm::vec3 Material::getNormal(float x, float y)
{
	if (normalTexture == 0)
		return NORMAL;
	else
		return glm::vec3(normalTexture->sampleTexture(x, y));
}

glm::vec4 Material::getAlbedo(float x, float y)
{
	if (albedoTexture == 0)
		return simpleAlbedo;
	else
		return albedoTexture->sampleTexture(x, y);
}

float Material::getRoughness(float x, float y)
{
	if (roughnessTexture == 0)
		return simpleRoughness.x;
	else
		return roughnessTexture->sampleTexture(x, y).x;
}

float Material::getMetal(float x, float y)
{
	if (metalTexture == 0)
		return simpleMetal.x;
	else
		return metalTexture->sampleTexture(x, y).x;
}

glm::vec3 Material::getAmbientOcclusion(float x, float y)
{
	if (ambientOcclusionTexture == 0)
		return AO;
	else
		return glm::vec3(ambientOcclusionTexture->sampleTexture(x, y));
}

glm::vec3 Material::getEmissionColor(float x, float y)
{
	if (emissionTexture == 0)
		return simpleEmissionColor;
	else
		return glm::vec3(emissionTexture->sampleTexture(x, y));
}

bool Material::setNormalTexture(const char* path)
{
	normalTexture = new Texture(path);
	return normalTexture->getLoadSuccess();
}

bool Material::setAlbedoTexture(const char* path)
{
	albedoTexture = new Texture(path);
	return albedoTexture->getLoadSuccess();
}

bool Material::setRoughnessTexture(const char* path)
{
	roughnessTexture = new Texture(path);
	return roughnessTexture->getLoadSuccess();
}

bool Material::setMetalTexture(const char* path)
{
	metalTexture = new Texture(path);
	return metalTexture->getLoadSuccess();
}

bool Material::setAmbientOcclusionTexture(const char* path)
{
	ambientOcclusionTexture = new Texture(path);
	return ambientOcclusionTexture->getLoadSuccess();
}

bool Material::setEmissionTexture(const char* path)
{
	emissionTexture = new Texture(path);
	return emissionTexture->getLoadSuccess();
}

Texture::Texture(const char* path)
{
	this->path = std::string(path);
	TextureMap::iterator it = textureMap.find(this->path);
	if (it != textureMap.end()) // Texture already loaded
	{
		it->second.first++;
		loadSuccess = it->second.second->loadSuccess;
		data = it->second.second->data;
		width = it->second.second->width;
		height = it->second.second->height;
		nChannels = it->second.second->nChannels;
		dataSize = it->second.second->dataSize;
	}
	else						// New texture
	{
		textureMap[this->path] = std::make_pair(1, this);
		unsigned char* idata = stbi_load(path, &width, &height, &nChannels, 0);
		if (!idata) {
			loadSuccess = false;
			return;
		}
		loadSuccess = true;
		dataSize = width * height * nChannels;
		data = new float[dataSize];
		for (int i = 0; i < dataSize; i++)
		{
			data[i] = (float)idata[i] / 255.0f;
		}
		stbi_image_free(idata);
	}
}

Texture::~Texture()
{
	if (--textureMap[path].first == 0)
	{
		delete[] data;
		textureMap.erase(path);
	}
}

// TODO: Implement bilinear interpolation
glm::vec4 Texture::sampleTexture(float x, float y)
{
	if (x < 0.0f || x > 1.0f || y < 0.0f || y > 1.0f)
		return glm::vec4(0.0f);
	int ix = (int)(x * width);
	int iy = (int)(y * height);
	
	int index = (iy * width + ix) * nChannels;
	if (nChannels == 3) 
		return glm::vec4(data[index], data[index + 1], data[index + 2], 1.0f);
	else
		return glm::vec4(data[index], data[index + 1], data[index + 2], data[index + 3]);
}




