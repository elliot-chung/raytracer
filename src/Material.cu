#include "Material.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

TextureMap Texture::textureMap = {};

MaterialMap Material::materialMap = {};

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

void Material::sendToGPU()
{
	if (!availableGPU) return;

	GPUMaterial hostCopy;
	hostCopy.albedo = make_float4(simpleAlbedo.r, simpleAlbedo.g, simpleAlbedo.b, simpleAlbedo.a);
	hostCopy.roughness = make_float3(simpleRoughness.r, simpleRoughness.g, simpleRoughness.b);
	hostCopy.metal = make_float3(simpleMetal.r, simpleMetal.g, simpleMetal.b);
	hostCopy.emissionColor = make_float3(simpleEmissionColor.r, simpleEmissionColor.g, simpleEmissionColor.b);
	hostCopy.emissionStrength = emissionStrength;

	if (albedoTexture != 0)
	{
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
		cudaArray* cuArray;
		checkCudaErrors(cudaMallocArray(&cuArray, &channelDesc, albedoTexture->getWidth(), albedoTexture->getHeight()));

		const size_t spitch = albedoTexture->getWidth() * sizeof(float4);
		checkCudaErrors(cudaMemcpy2DToArray(cuArray, 0, 0, albedoTexture->getData(), spitch, albedoTexture->getWidth() * sizeof(float4), albedoTexture->getHeight(), cudaMemcpyHostToDevice)); 
		
		struct cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = cuArray;

		struct cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc)); 
		texDesc.addressMode[0] = cudaAddressModeClamp; 
		texDesc.addressMode[1] = cudaAddressModeClamp; 
		texDesc.filterMode = cudaFilterModeLinear; 
		texDesc.readMode = cudaReadModeElementType; 
		texDesc.normalizedCoords = 1;

		checkCudaErrors(cudaCreateTextureObject(&hostCopy.albedoTexture, &resDesc, &texDesc, NULL));

	}
	else
		hostCopy.albedoTexture = 0;

	checkCudaErrors(cudaMalloc(&gpuMaterial, sizeof(GPUMaterial)));
	checkCudaErrors(cudaMemcpy(gpuMaterial, &hostCopy, sizeof(GPUMaterial), cudaMemcpyHostToDevice));
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
		// Convert to 4 channels
		{
			unsigned char* ndata = new unsigned char[width * height * 4];
			for (int i = 0; i < width * height; i++)
			{
				ndata[i * 4] = idata[i * nChannels];
				ndata[i * 4 + 1] = idata[i * nChannels + 1];
				ndata[i * 4 + 2] = idata[i * nChannels + 2];
				ndata[i * 4 + 3] = nChannels == 4 ? idata[i * nChannels + 3] : 255;
			}
			stbi_image_free(idata); 
			idata = ndata;
			nChannels = 4;
		}
		loadSuccess = true;
		dataSize = width * height * nChannels;
		data = new float[dataSize];
		for (int i = 0; i < dataSize; i++)
		{
			data[i] = (float)idata[i] / 255.0f;
		}
		delete[] idata;
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

__device__ float3 GPUMaterial::getNormal(float x, float y)  
{
	if (normalTexture == 0)
		return NORMAL;
	else
	{
		float4 n = tex2D<float4>(normalTexture, x, y);
		return make_float3(n.x, n.y, n.z);
	}
}

__device__ float4 GPUMaterial::getAlbedo(float x, float y)
{
	if (albedoTexture == 0)
		return albedo;
	else
		return tex2D<float4>(albedoTexture, x, y);
}

__device__ float GPUMaterial::getRoughness(float x, float y)
{
	if (roughnessTexture == 0)
		return roughness.x;
	else
		return tex2D<float4>(roughnessTexture, x, y).x;
}

__device__ float GPUMaterial::getMetal(float x, float y)
{
	if (metalTexture == 0)
		return metal.x;
	else
		return tex2D<float4>(metalTexture, x, y).x;
}

__device__ float3 GPUMaterial::getAmbientOcclusion(float x, float y)
{
	if (ambientOcclusionTexture == 0)
		return AO;
	else
	{
		float4 ao = tex2D<float4>(ambientOcclusionTexture, x, y);
		return make_float3(ao.x, ao.y, ao.z);
	}
}

__device__ float4 GPUMaterial::getEmission(float x, float y)
{
	if (emissionTexture == 0)
		return make_float4(emissionColor.x, emissionColor.y, emissionColor.z, emissionStrength);
	else
	{
		float4 e = tex2D<float4>(emissionTexture, x, y);
		return make_float4(e.x, e.y, e.z, emissionStrength);
	}
}










