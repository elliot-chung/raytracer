#include "GPURaytracer.hpp"

// Credit https://github.com/SebLague/Ray-Tracing/blob/main/Assets/Scripts/Shaders/RayTracing.shader
__device__ float4 getSkyLight(const float4& direction, const float4& lightDirection, const float4& lightColor, const float4& skyColor) 
{
	/*float dotProduct = dot(direction, lightDirection);
	if (1.0f - dotProduct < 0.01f)
		return lightColor;
	else
		return skyColor;*/

	// float skyGradientT = powf(smoothstep(0, 0.4, direction.y), 0.35);
	// float groundToSkyT = smoothstep(-0.01, 0, direction.y);
	// float3 skyGradient = lerp(SkyColourHorizon, SkyColourZenith, skyGradientT);
	float sun = powf(max(0.0f, dot(direction, lightDirection)), 1000) * lightColor.w;
	
	return make_float4(skyColor.x * skyColor.w + lightColor.x * sun,
					   skyColor.y * skyColor.w + lightColor.y * sun,
		               skyColor.z * skyColor.w + lightColor.z * sun, 1.0f); 
}



void GPURaytracer::raytrace(std::shared_ptr<Scene> scene, std::shared_ptr<Camera> camera, cudaSurfaceObject_t canvas)
{
	// Camera Parameters
	CameraParams camSettings = {};
	glm::vec4 position = glm::vec4(camera->getPosition(), 1.0f);
	glm::quat rotation = camera->getRotation();
	float exposure = camera->getExposure();
	int width = camera->getWidth();
	int height = camera->getHeight();
	cudaTextureObject_t rays = camera->getCudaRays();

	vec4transfer(camSettings.origin, position);
	vec4transfer(camSettings.rotation, rotation);
	camSettings.exposure = exposure;
	camSettings.width = width;
	camSettings.height = height;
	camSettings.rays = rays;

	// Raytracing Renderer Parameters
	RendererParams rendererSettings = {};
	rendererSettings.bounceCount = bounceCount;
	rendererSettings.maxDistance = maxDistance;
	rendererSettings.aoIntensity = aoIntensity;
	rendererSettings.frameCount = frameCount;
	rendererSettings.progressiveFrameCount = progressiveFrameCount;
	rendererSettings.antiAliasingEnabled = antiAliasingEnabled;
	rendererSettings.sampleCount = sampleCount;

	SkyLightParams skyLightSettings = {}; 
	skyLightSettings.direction = lightDirection;
	skyLightSettings.lightColor = lightColor;
	skyLightSettings.skyColor = skyColor;

	// -----------------------------
	// Transfer Scene Data to GPU
	Scene::ObjectMap objects = scene->getObjects();
	ObjectData* objectDataArray = new ObjectData[objects.size()];
	int i = 0;
	for (auto& objPair : objects)
	{
		DisplayObject* obj = objPair.second;
		ObjectData data = {};

		mat4transfer(data.modelMatrix, obj->getModelMatrix());
		data.mesh = obj->getMesh()->getGPUMeshData();
		data.material = obj->getGPUMaterial();
		objectDataArray[i++] = data;
	}
	ObjectData* objectDataArrayDev;
	checkCudaErrors(cudaMalloc((void**)&objectDataArrayDev, sizeof(ObjectData) * objects.size()));
	checkCudaErrors(cudaMemcpy(objectDataArrayDev, objectDataArray, sizeof(ObjectData) * objects.size(), cudaMemcpyHostToDevice));

	ObjectDataVector objectDataVector = {};
	objectDataVector.data = objectDataArrayDev;
	objectDataVector.size = objects.size();

	delete[] objectDataArray;
	// (This block of code should be moved to the Scene class, remember to move the cudaFree call as well)
	// ----------------------------

	// Create Debug Output
	DebugInfo* debugInfo = nullptr;
	if (debug)
	{
		checkCudaErrors(cudaMalloc((void**)&debugInfo, sizeof(DebugInfo)));
		checkCudaErrors(cudaMemset(debugInfo, -1, sizeof(DebugInfo)));
	}

	// Launch Kernel
	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, MAXIMUM_AA); 
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

	raytraceKernel<<<gridSize, blockSize>>>(camSettings, canvas, objectDataVector, rendererSettings, skyLightSettings, debug, debugInfo);
	checkCudaErrors(cudaPeekAtLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Transfer Debug Output
	if (debug) 
	{
		DebugInfo* debugInfoHost = new DebugInfo;
		checkCudaErrors(cudaMemcpy(debugInfoHost, debugInfo, sizeof(DebugInfo), cudaMemcpyDeviceToHost));
		std::cout << "First Hit Object:\t" << debugInfoHost->firstObjectDataIndex << '\t';
		std::cout << "First Hit Origin:\t" << debugInfoHost->firstOrigin.x << ", " << debugInfoHost->firstOrigin.y << ", " << debugInfoHost->firstOrigin.z << '\t';
		std::cout << "First Hit Direction:\t" << debugInfoHost->firstDirection.x << ", " << debugInfoHost->firstDirection.y << ", " << debugInfoHost->firstDirection.z << '\t';
		std::cout << "First Hit Position:\t" << debugInfoHost->firstPosition.x << ", " << debugInfoHost->firstPosition.y << ", " << debugInfoHost->firstPosition.z << '\t';
		std::cout << "First Hit Normal:\t" << debugInfoHost->firstNormal.x << ", " << debugInfoHost->firstNormal.y << ", " << debugInfoHost->firstNormal.z << '\t';
		std::cout << "First Hit Distance:\t" << debugInfoHost->firstDistance << "\n" << std::endl;

		std::cout << "Second Hit Object:\t" << debugInfoHost->secondObjectDataIndex << '\t';
		std::cout << "Second Hit Origin:\t" << debugInfoHost->secondOrigin.x << ", " << debugInfoHost->secondOrigin.y << ", " << debugInfoHost->secondOrigin.z << '\t';
		std::cout << "Second Hit Direction:\t" << debugInfoHost->secondDirection.x << ", " << debugInfoHost->secondDirection.y << ", " << debugInfoHost->secondDirection.z << '\t';
		std::cout << "Second Hit Position:\t" << debugInfoHost->secondPosition.x << ", " << debugInfoHost->secondPosition.y << ", " << debugInfoHost->secondPosition.z << '\t';
		std::cout << "Second Hit Normal:\t" << debugInfoHost->secondNormal.x << ", " << debugInfoHost->secondNormal.y << ", " << debugInfoHost->secondNormal.z << '\t';
		std::cout << "Second Hit Distance:\t" << debugInfoHost->secondDistance << "\n" << std::endl;
	}
	
	frameCount++;
	if (progressiveRendering)
		progressiveFrameCount++;
	else 
		progressiveFrameCount = 0;
	

	checkCudaErrors(cudaFree(objectDataArrayDev));
	if (debug) checkCudaErrors(cudaFree(debugInfo));
}



__global__ void raytraceKernel(CameraParams camera, cudaSurfaceObject_t canvas, ObjectDataVector objectDataVector, const RendererParams renderer, const SkyLightParams skylight, const bool debug, DebugInfo* debugInfo)
{
	__shared__ float4 sharedMemory[BLOCK_SIZE][BLOCK_SIZE][MAXIMUM_AA]; 
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= camera.width || y >= camera.height || threadIdx.z >= renderer.sampleCount)
		return;

	float4 color = make_float4(0.0f, 0.0f, 0.0f, 1.0f); 
	for (int iter = 0; iter < (renderer.sampleCount + MAXIMUM_AA - 1) / MAXIMUM_AA; iter++) 
	{
		if (iter * MAXIMUM_AA + threadIdx.z >= renderer.sampleCount) break; // Early exit (if sampleCount is not a multiple of MAXIMUM_AA)

		bool isCenter = x == camera.width / 2 && y == camera.height / 2 && threadIdx.z == 0 && iter == 0;

		unsigned int seed = (x + y * camera.width + (iter * MAXIMUM_AA + threadIdx.z) * 34673804 + renderer.frameCount * 719393);

		GPURay ray = setupRay(camera, x, y, renderer.bounceCount, renderer.maxDistance, renderer.antiAliasingEnabled, seed);

		float4 partialColor = trace(ray, objectDataVector, skylight, renderer.aoIntensity, seed, (debug && isCenter), debugInfo);

		partialColor = exposureCorrection(partialColor, camera.exposure);

		color.x += partialColor.x;
		color.y += partialColor.y;
		color.z += partialColor.z;
	}

	sharedMemory[threadIdx.x][threadIdx.y][threadIdx.z] = color;
	__syncthreads();

	if (threadIdx.z == 0) {
		color = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
		for (int i = 0; i < MAXIMUM_AA; i++) {
			color.x += sharedMemory[threadIdx.x][threadIdx.y][i].x;
			color.y += sharedMemory[threadIdx.x][threadIdx.y][i].y;
			color.z += sharedMemory[threadIdx.x][threadIdx.y][i].z;
		}  
		color.x /= renderer.sampleCount;
		color.y /= renderer.sampleCount;
		color.z /= renderer.sampleCount;
	} else {
		return;
	}

	// Clamp color values
	color.x = fminf(fmaxf(color.x, 0.0f), 1.0f);
	color.y = fminf(fmaxf(color.y, 0.0f), 1.0f);
	color.z = fminf(fmaxf(color.z, 0.0f), 1.0f);

	if (renderer.progressiveFrameCount != 0) {
		float4 prevColor = surf2Dread<float4>(canvas, x * sizeof(float4), y);
		color.x = (color.x + prevColor.x * renderer.progressiveFrameCount) / (renderer.progressiveFrameCount + 1);
		color.y = (color.y + prevColor.y * renderer.progressiveFrameCount) / (renderer.progressiveFrameCount + 1);
		color.z = (color.z + prevColor.z * renderer.progressiveFrameCount) / (renderer.progressiveFrameCount + 1);
	} 

	
	surf2Dwrite(color, canvas, x * sizeof(float4), y);
}

__device__ GPURay setupRay(const CameraParams& camera, const int x, const int y, const int bounceCount, const float maxDistance, const bool aaEnabled, unsigned int& seed)
{
	GPURay ray = {};
	ray.origin = camera.origin; 
	ray.bounceCount = bounceCount; 
	ray.maxDistance = maxDistance; 

	if (aaEnabled)
	{
		float xf = (float)x + randomValue(seed) - 0.5f;
		float yf = (float)y + randomValue(seed) - 0.5f;

		ray.direction = tex2D<float4>(camera.rays, xf, yf);
	}
	else
	{
		ray.direction = tex2D<float4>(camera.rays, x, y);
	}
	
	ray.direction = rotate(ray.direction, camera.rotation); 
	ray.direction = normalize(ray.direction); 

	return ray;
}

__device__ float4 trace(GPURay& ray, const ObjectDataVector& objectDataVector, SkyLightParams skylight, const float aoIntensity, unsigned int& seed, const bool debug, DebugInfo* debugInfo)
{
	float4 outgoingLight = make_float4(0.0f, 0.0f, 0.0f, 1.0f);  
	float4 betaAccumulation = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
	
	// Bounce Loop
	// i - bounce iteration (each iteration is a bounce)
	for (int i = 0; i < ray.bounceCount; i++)
	{
		// Find ray object intersection and populate material data accodingly (break out on miss)
		GPURayHit closestHit = getIntersectionPoint(ray, objectDataVector); 
		if (!closestHit.didHit)
		{
			float4 missColor = getSkyLight(ray.direction, skylight.direction, skylight.lightColor, skylight.skyColor);
			outgoingLight.x += missColor.x * missColor.w * betaAccumulation.x;
			outgoingLight.y += missColor.y * missColor.w * betaAccumulation.y;
			outgoingLight.z += missColor.z * missColor.w * betaAccumulation.z;
			break;
		}
		GPUMaterialPositionData materialData = getMaterialData(closestHit); 

		// Populate Debug info based on hit information
		if (debug)
		{
			if (i == 0) 
			{
				debugInfo->firstObjectDataIndex = closestHit.objectDataIndex;
				debugInfo->firstOrigin = ray.origin;
				debugInfo->firstDirection = ray.direction;
				debugInfo->firstPosition = closestHit.hitPosition;
				debugInfo->firstNormal = materialData.normal;
				debugInfo->firstDistance = closestHit.distance;
			} else if (i == 1)
			{
				debugInfo->secondObjectDataIndex = closestHit.objectDataIndex;
				debugInfo->secondOrigin = ray.origin;
				debugInfo->secondDirection = ray.direction;
				debugInfo->secondPosition = closestHit.hitPosition;
				debugInfo->secondNormal = materialData.normal;
				debugInfo->secondDistance = closestHit.distance;
			}
		}
		
		// Outgoing light only increases on the first hit (ambient occlusion) or when hitting an emissive material
		if (i == 0) // First Hit
		{
			// Ambient Occlusion
			outgoingLight.x += materialData.ao.x * materialData.albedo.x * aoIntensity; 
			outgoingLight.y += materialData.ao.y * materialData.albedo.y * aoIntensity; 
			outgoingLight.z += materialData.ao.z * materialData.albedo.z * aoIntensity; 
		}

		// Emission
		float4 emittedLight = materialData.emission;
		outgoingLight.x += emittedLight.x * emittedLight.w * betaAccumulation.x;  
		outgoingLight.y += emittedLight.y * emittedLight.w * betaAccumulation.y;  
		outgoingLight.z += emittedLight.z * emittedLight.w * betaAccumulation.z;  

		// Cook-torrance BRDF credit: https://learnopengl.com/PBR/Lighting
		if (i < ray.bounceCount - 1) // Only if there's another iteration after this one
		{
			// float attenuation = 1.0f / (closestHit.distance * closestHit.distance); 
			float attenuation = 1.0f;
			float4 N = materialData.normal;
			float4 diffuseDirection = randomUnitVectorInCosineHemisphere(seed, N);  
			float4 specularDirection = reflect(ray.direction, N);
			float4 L = normalize(lerp(specularDirection, diffuseDirection, materialData.roughness));  
			float4 V = negate(ray.direction);
			float4 H = normalize(make_float4(L.x + V.x, L.y + V.y, L.z + V.z, 0.0f));

			float4 F0 = make_float4(0.04f, 0.04f, 0.04f, 1.0f);
			F0 = lerp(F0, materialData.albedo, materialData.metal);
			
			float NDF = distributionGGX(N, H, materialData.roughness);
			float G = geometrySmith(N, V, L, materialData.roughness);
			float4 F = fresnelSchlick(max(dot(H, V), 0.0f), F0); 

			float4 kD = make_float4(1.0f - F.x, 1.0f - F.y, 1.0f - F.z, 1.0f);
			float metalComplement = 1.0f - materialData.metal;
			kD.x = kD.x * metalComplement;
			kD.y = kD.y * metalComplement;	
			kD.z = kD.z * metalComplement;

			
			float4 numerator = make_float4(NDF * G * F.x, NDF * G * F.y, NDF * G * F.z, 1.0f);
			float denominator = 4.0f * max(dot(N, V), 0.0f) * max(dot(N, L), 0.0f) + 0.0001f;
			float4 specular = make_float4(numerator.x / denominator, numerator.y / denominator, numerator.z / denominator, 1.0f);

			// Accumulate Color data
			float NdotL = max(dot(N, L), 0.0f);
			betaAccumulation.x *= (kD.x * materialData.albedo.x / PI + specular.x) * attenuation * NdotL; 
			betaAccumulation.y *= (kD.y * materialData.albedo.y / PI + specular.y) * attenuation * NdotL;
			betaAccumulation.z *= (kD.z * materialData.albedo.z / PI + specular.z) * attenuation * NdotL;

			// Update Ray
			float nudgeStrength = 0.0001f;
			ray.origin = make_float4(closestHit.hitPosition.x + N.x * nudgeStrength, closestHit.hitPosition.y + N.y * nudgeStrength, closestHit.hitPosition.z + N.z * nudgeStrength, 1.0f);
			ray.direction = L; 
		}
	} 

	return outgoingLight;
}

__device__ GPURayHit getIntersectionPoint(GPURay& ray, const ObjectDataVector& dataVector)
{
	GPURayHit closestHit = {};
	closestHit.distance = FLT_MAX;
	for (int i = 0; i < dataVector.size; i++)
	{
		ObjectData data = dataVector.data[i];
		if (!intersectsBoundingBox(ray, data)) continue;
		GPURayHit hp = getIntersectionPoint(ray, data);
		if (hp.didHit && hp.distance < closestHit.distance)
		{
			closestHit = hp;
			closestHit.objectDataIndex = i;

			ray.maxDistance = hp.distance;
		}
	}
	return closestHit;
}

__device__ bool intersectsBoundingBox(const GPURay& ray, const ObjectData& data) 
{
	float tmin = 0.0f;
	float tmax = ray.maxDistance;

	float4 minBound = make_float4(data.mesh->minBounds.x, data.mesh->minBounds.y, data.mesh->minBounds.z, 1.0f);
	float4 maxBound = make_float4(data.mesh->maxBounds.x, data.mesh->maxBounds.y, data.mesh->maxBounds.z, 1.0f);

	minBound = matVecMul(data.modelMatrix, minBound);
	maxBound = matVecMul(data.modelMatrix, maxBound);

	float3 dirInv = make_float3(1.0f / ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z); 

	bool sign = signbit(ray.direction.x); 
	float bmin = sign ? maxBound.x : minBound.x; 
	float bmax = sign ? minBound.x : maxBound.x; 

	float dmin = (bmin - ray.origin.x) * dirInv.x; 
	float dmax = (bmax - ray.origin.x) * dirInv.x; 

	tmin = max(dmin, tmin);  
	tmax = min(dmax, tmax);  
	 
	sign = signbit(ray.direction.y); 
	bmin = sign ? maxBound.y : minBound.y; 
	bmax = sign ? minBound.y : maxBound.y; 

	dmin = (bmin - ray.origin.y) * dirInv.y; 
	dmax = (bmax - ray.origin.y) * dirInv.y;  

	tmin = max(dmin, tmin); 
	tmax = min(dmax, tmax);  

	sign = signbit(ray.direction.z); 
	bmin = sign ? maxBound.z : minBound.z; 
	bmax = sign ? minBound.z : maxBound.z;  

	dmin = (bmin - ray.origin.z) * dirInv.z;  
	dmax = (bmax - ray.origin.z) * dirInv.z;  
	 
	tmin = max(dmin, tmin);  
	tmax = min(dmax, tmax);  

	return tmin < tmax;

}

__device__  GPUMaterialPositionData getMaterialData(const GPURayHit& hit)
{
	GPUMaterialPositionData output = {};
	float2 uv = hit.uv;
	mat3 tbnMatrix = hit.tbnMatrix;

	output.albedo = hit.material->getAlbedo(uv.x, uv.y);
	float3 normal3 = normalize(matVecMul(tbnMatrix, hit.material->getNormal(uv.x, uv.y))); 
	output.normal = make_float4(normal3.x, normal3.y, normal3.z, 0.0f);
	output.roughness = hit.material->getRoughness(uv.x, uv.y);
	output.metal = hit.material->getMetal(uv.x, uv.y);
	output.ao = hit.material->getAmbientOcclusion(uv.x, uv.y);
	output.emission = hit.material->getEmission(uv.x, uv.y);
	
	return output;
}

__device__ GPURayHit getIntersectionPoint(const GPURay& ray, const ObjectData& data)
{
	mat4 modelMatrix = data.modelMatrix;
	GPUMeshData* meshData = data.mesh;
	float* vertices = meshData->vertices;
	float* uvCoords = meshData->uvs;
	int* indices = meshData->indices;
	float* normals = meshData->normals;
	
	int closestTriangle = -1;
	GPUTriangleHit closestHit = {};
	closestHit.distance = FLT_MAX;

	for (int i = 0; i < meshData->triangleCount; i++)
	{
		int i0 = indices[i * 3 + 0] * 3;
		int i1 = indices[i * 3 + 1] * 3;
		int i2 = indices[i * 3 + 2] * 3;

		float4 v0 = make_float4( vertices[i0 + 0], vertices[i0 + 1], vertices[i0 + 2], 1.0f);
		float4 v1 = make_float4( vertices[i1 + 0], vertices[i1 + 1], vertices[i1 + 2], 1.0f);
		float4 v2 = make_float4( vertices[i2 + 0], vertices[i2 + 1], vertices[i2 + 2], 1.0f);

		v0 = matVecMul(modelMatrix, v0);
		v1 = matVecMul(modelMatrix, v1);
		v2 = matVecMul(modelMatrix, v2);

		GPUTriangleHit triHit = distToTriangle(ray, v0, v1, v2);
		if (triHit.distance < closestHit.distance)
		{
			closestHit = triHit;
			closestTriangle = i;
		}
	}

	GPURayHit output = {};
	output.didHit = false;

	if (closestTriangle == -1) return output;
	
	int i0 = indices[closestTriangle * 3 + 0];
	int i1 = indices[closestTriangle * 3 + 1];
	int i2 = indices[closestTriangle * 3 + 2];

	int i0uv = i0 * 2;
	int i1uv = i1 * 2;
	int i2uv = i2 * 2;

	i0 *= 3;
	i1 *= 3;
	i2 *= 3;

	float4 v0 = make_float4(vertices[i0 + 0], vertices[i0 + 1], vertices[i0 + 2], 1.0f); 
	float4 v1 = make_float4(vertices[i1 + 0], vertices[i1 + 1], vertices[i1 + 2], 1.0f); 
	float4 v2 = make_float4(vertices[i2 + 0], vertices[i2 + 1], vertices[i2 + 2], 1.0f); 


	float4 n0;
	float4 n1;
	float4 n2;	
	if (normals) 
	{
		n0 = make_float4(normals[i0 + 0], normals[i0 + 1], normals[i0 + 2], 0.0f);
		n1 = make_float4(normals[i1 + 0], normals[i1 + 1], normals[i1 + 2], 0.0f);
		n2 = make_float4(normals[i2 + 0], normals[i2 + 1], normals[i2 + 2], 0.0f);
	}

	v0 = matVecMul(modelMatrix, v0);
	v1 = matVecMul(modelMatrix, v1);
	v2 = matVecMul(modelMatrix, v2);

	float2 uv0 = make_float2(uvCoords[i0uv + 0], uvCoords[i0uv + 1]); 
	float2 uv1 = make_float2(uvCoords[i1uv + 0], uvCoords[i1uv + 1]); 
	float2 uv2 = make_float2(uvCoords[i2uv + 0], uvCoords[i2uv + 1]); 

	float4 edge1 = make_float4(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z, 0.0f); 
	float4 edge2 = make_float4(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z, 0.0f); 
	float2 deltaUV1 = make_float2(uv1.x - uv0.x, uv1.y - uv0.y); 
	float2 deltaUV2 = make_float2(uv2.x - uv0.x, uv2.y - uv0.y); 

	float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);
	float4 tangent = make_float4(
		f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x),
		f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y),
		f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z), 
		0.0f
	);
	float4 bitangent = make_float4(
		f * (-deltaUV2.x * edge1.x + deltaUV1.x * edge2.x), 
		f * (-deltaUV2.x * edge1.y + deltaUV1.x * edge2.y), 
		f * (-deltaUV2.x * edge1.z + deltaUV1.x * edge2.z), 
		0.0f
	);
	tangent = normalize(tangent); 
	bitangent = normalize(bitangent);
	float4 normal; 
	if (normals)
	{
		normal = make_float4(
			n0.x * closestHit.barycentricCoords.x + n1.x * closestHit.barycentricCoords.y + n2.x * closestHit.barycentricCoords.z,
			n0.y * closestHit.barycentricCoords.x + n1.y * closestHit.barycentricCoords.y + n2.y * closestHit.barycentricCoords.z,
			n0.z * closestHit.barycentricCoords.x + n1.z * closestHit.barycentricCoords.y + n2.z * closestHit.barycentricCoords.z,
			0.0f);
		// Gram-Schmidt orthogonalize
		{
			float correction = dot(tangent, normal);
			tangent = normalize(make_float4(
				tangent.x - correction * normal.x,
				tangent.y - correction * normal.y,
				tangent.z - correction * normal.z,
				0.0f));
			bitangent = normalize(cross(normal, tangent));
		}
	}
	else
		normal = normalize(cross(edge2, edge1));  

	mat3 tbnMatrix = {};
	tbnMatrix.c0 = make_float3(tangent.x, tangent.y, tangent.z);   
	tbnMatrix.c1 = make_float3(bitangent.x, bitangent.y, bitangent.z);   
	tbnMatrix.c2 = make_float3(normal.x, normal.y, normal.z);  

	float4 interpPosition = make_float4( 
		v0.x * closestHit.barycentricCoords.x + v1.x * closestHit.barycentricCoords.y + v2.x * closestHit.barycentricCoords.z,
		v0.y * closestHit.barycentricCoords.x + v1.y * closestHit.barycentricCoords.y + v2.y * closestHit.barycentricCoords.z,
		v0.z * closestHit.barycentricCoords.x + v1.z * closestHit.barycentricCoords.y + v2.z * closestHit.barycentricCoords.z,
		1.0f
	);
	float2 interpUV = make_float2(
		uv0.x * closestHit.barycentricCoords.x + uv1.x * closestHit.barycentricCoords.y + uv2.x * closestHit.barycentricCoords.z,
		uv0.y * closestHit.barycentricCoords.x + uv1.y * closestHit.barycentricCoords.y + uv2.y * closestHit.barycentricCoords.z
	);

	output.didHit = true;
	output.distance = closestHit.distance;
	output.hitPosition = interpPosition;
	output.uv = interpUV;
	output.tbnMatrix = tbnMatrix;
	output.material = data.material;

	return output;
}

__device__ GPUTriangleHit distToTriangle(const GPURay& ray, const float4& v0, const float4& v1, const float4& v2)
{
	GPUTriangleHit output = {};
	output.distance = FLT_MAX;
	output.barycentricCoords = make_float3(0.0f, 0.0f, 0.0f);

	float3 v0t = make_float3(v0.x - ray.origin.x, v0.y - ray.origin.y, v0.z - ray.origin.z);
	float3 v1t = make_float3(v1.x - ray.origin.x, v1.y - ray.origin.y, v1.z - ray.origin.z);
	float3 v2t = make_float3(v2.x - ray.origin.x, v2.y - ray.origin.y, v2.z - ray.origin.z);

	float3 d = make_float3(ray.direction.x, ray.direction.y, ray.direction.z);

	float absDirx = fabsf(ray.direction.x);
	float absDiry = fabsf(ray.direction.y);
	float absDirz = fabsf(ray.direction.z);
	int kz = absDirx > absDiry ? (absDirx > absDirz ? 0 : 2) : (absDiry > absDirz ? 1 : 2); 
	int kx = kz == 2 ? 0 : kz + 1; 
	int ky = kx == 2 ? 0 : kx + 1; 

	d = make_float3(((float*) &d)[kx], ((float*)&d)[ky], ((float*)&d)[kz]); 
	v0t = make_float3(((float*) &v0t)[kx], ((float*)&v0t)[ky], ((float*)&v0t)[kz]); 
	v1t = make_float3(((float*)&v1t)[kx], ((float*)&v1t)[ky], ((float*)&v1t)[kz]);
	v2t = make_float3(((float*)&v2t)[kx], ((float*)&v2t)[ky], ((float*)&v2t)[kz]); 

	float sz = 1.f / d.z; 
	float sx = -d.x * sz; 
	float sy = -d.y * sz; 

	v0t.x += sx * v0t.z; 
	v0t.y += sy * v0t.z; 
	v1t.x += sx * v1t.z; 
	v1t.y += sy * v1t.z; 
	v2t.x += sx * v2t.z; 
	v2t.y += sy * v2t.z; 

	float e0 = v1t.x * v2t.y - v1t.y * v2t.x; 
	float e1 = v2t.x * v0t.y - v2t.y * v0t.x; 
	float e2 = v0t.x * v1t.y - v0t.y * v1t.x; 

	if (e0 == 0.0f || e1 == 0.0f || e2 == 0.0f)
	{
		double p2txp1ty = (double)v2t.x * (double)v1t.y; 
		double p2typ1tx = (double)v2t.y * (double)v1t.x; 
		e0 = (float)(p2typ1tx - p2txp1ty); 
		double p0txp2ty = (double)v0t.x * (double)v2t.y; 
		double p0typ2tx = (double)v0t.y * (double)v2t.x; 
		e1 = (float)(p0typ2tx - p0txp2ty);
		double p1txp0ty = (double)v1t.x * (double)v0t.y;
		double p1typ0tx = (double)v1t.y * (double)v0t.x;
		e2 = (float)(p1typ0tx - p1txp0ty);
	}

	if ((e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0))
		return output;
	float det = e0 + e1 + e2;
	if (det == 0)
		return output;

	// Check point distance from origin
	v0t.z *= sz; 
	v1t.z *= sz; 
	v2t.z *= sz; 
	float tScaled = e0 * v0t.z + e1 * v1t.z + e2 * v2t.z; 
	if (det < 0 && (tScaled >= 0 || tScaled < ray.maxDistance * det)) 
		return output;
	else if (det > 0 && (tScaled <= 0 || tScaled > ray.maxDistance * det))
		return output;


	// Calculate barycentric coords and parametric value (distance)   
	float invDet = 1 / det;
	float b0 = e0 * invDet;
	float b1 = e1 * invDet;
	float b2 = e2 * invDet;
	float t = tScaled * invDet;

	if (t < 0.001 || t > ray.maxDistance)
		return output;

	
	output.distance = t;
	output.barycentricCoords.x = b0;
	output.barycentricCoords.y = b1;
	output.barycentricCoords.z = b2;
	
	return output;
}

// Credit https://learnopengl.com/PBR/Lighting
__device__ float4 fresnelSchlick(const float cosTheta, const float4& f0)
{
	return make_float4(
		f0.x + (1.0f - f0.x) * powf(1.0f - min(cosTheta, 1.0f), 5.0f),
		f0.y + (1.0f - f0.y) * powf(1.0f - min(cosTheta, 1.0f), 5.0f),
		f0.z + (1.0f - f0.z) * powf(1.0f - min(cosTheta, 1.0f), 5.0f), 
		1.0f
	);
}

// Credit https://learnopengl.com/PBR/Lighting 
__device__ float distributionGGX(const float4& n, const float4& h, const float roughness)
{
	float a = roughness * roughness;
	float a2 = a * a;
	float NdotH = max(0.0f, dot(n, h));
	float NdotH2 = NdotH * NdotH;

	float num = a2;
	float denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
	denom = PI * denom * denom;

	return num / (denom + 0.00001f);
}

// Credit https://learnopengl.com/PBR/Lighting
__device__ float geometrySchlickGGX(const float NdotV, const float roughness)
{
	float r = (roughness + 1.0f);
	float k = (r * r) / 8.0f;

	float num = NdotV;
	float denom = NdotV * (1.0f - k) + k;

	return num / denom;
} 

// Credit https://learnopengl.com/PBR/Lighting
__device__ float geometrySmith(const float4& n, const float4& v, const float4& l, const float roughness)
{
	float NdotV = max(dot(n, v), 0.0f);
	float NdotL = max(dot(n, l), 0.0f);
	float ggx1 = geometrySchlickGGX(NdotV, roughness);
	float ggx2 = geometrySchlickGGX(NdotL, roughness);

	return ggx1 * ggx2;
} 



__device__ __forceinline__ float4 exposureCorrection(const float4 color, const float exposure)
{
	return make_float4(
		1.0f - expf(-color.x * exposure),
		1.0f - expf(-color.y * exposure),
		1.0f - expf(-color.z * exposure),
		1.0f
	);
}

__device__ __forceinline__ float4 rotate(const float4 v, const float4 q)
{
	float t2 = q.w * q.x;
	float t3 = q.w * q.y;
	float t4 = q.w * q.z;
	float t5 = -q.x * q.x;
	float t6 = q.x * q.y;
	float t7 = q.x * q.z;
	float t8 = -q.y * q.y;
	float t9 = q.y * q.z;
	float t10 = -q.z * q.z;
	return make_float4(
		2.0f * ((t8 + t10) * v.x + (t6 - t4) * v.y + (t3 + t7) * v.z) + v.x,
		2.0f * ((t4 + t6) * v.x + (t5 + t10) * v.y + (t9 - t2) * v.z) + v.y,
		2.0f * ((t7 - t3) * v.x + (t2 + t9) * v.y + (t5 + t8) * v.z) + v.z, 
		v.w
	);
}

__device__ __forceinline__ float3 matVecMul(const mat3 m, const float3 v)
{
	return make_float3(
		m.c0.x * v.x + m.c1.x * v.y + m.c2.x * v.z, 
		m.c0.y * v.x + m.c1.y * v.y + m.c2.y * v.z, 
		m.c0.z * v.x + m.c1.z * v.y + m.c2.z * v.z 
	);
}

__device__ __forceinline__ float4 matVecMul(const mat4 m, const float4 v)
{
	return make_float4(
		m.c0.x * v.x + m.c1.x * v.y + m.c2.x * v.z + m.c3.x * v.w,
		m.c0.y * v.x + m.c1.y * v.y + m.c2.y * v.z + m.c3.y * v.w,
		m.c0.z * v.x + m.c1.z * v.y + m.c2.z * v.z + m.c3.z * v.w,
		m.c0.w * v.x + m.c1.w * v.y + m.c2.w * v.z + m.c3.w * v.w
	);
}

__device__ __forceinline__ float4 cross(const float4 a, const float4 b)
{
	return make_float4(
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z, 
		a.x * b.y - a.y * b.x, 
		0.0f
	);
}

__device__ __forceinline__ float dot(const float4 a, const float4 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float4 negate(const float4 v)
{
	return make_float4(-v.x, -v.y, -v.z, v.w);
}

__device__ __forceinline__ float3 negate(const float3 v)
{
	return make_float3(-v.x, -v.y, -v.z);
}

__device__ __forceinline__ float4 lerp(const float4 a, const float4 b, const float t)
{
	float4 output = make_float4(
		a.x + (b.x - a.x) * t,
		a.y + (b.y - a.y) * t,
		a.z + (b.z - a.z) * t,
		a.w + (b.w - a.w) * t
	);
	return output;
}

__device__ __forceinline__ float4 normalize(const float4 v)
{
	float invLength = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
	return make_float4(v.x * invLength, v.y * invLength, v.z * invLength, v.w);
}

__device__ __forceinline__ float3 normalize(const float3 v)
{
	float invLength = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
	return make_float3(v.x * invLength, v.y * invLength, v.z * invLength);
}

__device__ __forceinline__ float randomValue(unsigned int& seed)
{
	seed = seed * 747796405 + 2891336453;
	unsigned int result = ((seed >> ((seed >> 28) + 4)) ^ seed) * 277803737;
	result = (result >> 22) ^ result;
	return result / 4294967295.0;
}

__device__ __forceinline__ float randomValueNormalDistribution(unsigned int& seed)
{
	float theta = 2 * PI * randomValue(seed);
	float r = randomValue(seed);
	float rho = sqrt(-2 * log(r));
	return rho * cos(theta);
}

__device__ __forceinline__ float4 randomUnitVector(unsigned int& seed)
{
	float x = randomValueNormalDistribution(seed); 
	float y = randomValueNormalDistribution(seed); 
	float z = randomValueNormalDistribution(seed); 
	return normalize(make_float4(x, y, z, 0.0f));
}

__device__ __forceinline__ float4 randomUnitVectorInHemisphere(unsigned int& seed, const float4& normal)
{
	float4 unitVector = randomUnitVector(seed);

	if (dot(unitVector, normal) > 0.0f)
		return unitVector;
	else
		return negate(unitVector);
}

__device__ __forceinline__ float4 randomUnitVectorInCosineHemisphere(unsigned int& seed, const float4& normal)
{
	float4 unitVector = randomUnitVector(seed);
	float4 output = make_float4(unitVector.x + normal.x, unitVector.y + normal.y, unitVector.z + normal.z, 0.0f);
	output = normalize(output);
	return output;
}

__device__ __forceinline__ float4 reflect(const float4& v, const float4& normal)
{
	float dotProduct = dot(v, normal);
	float4 output = make_float4(
		v.x - 2.0f * dotProduct * normal.x,
		v.y - 2.0f * dotProduct * normal.y,
		v.z - 2.0f * dotProduct * normal.z,
		0.0f
	);
	output = normalize(output);
	return output;
}

__device__ __forceinline__ float smoothstep(const float edge0, const float edge1, float x)
{
	// Scale, and clamp x to 0..1 range
	x = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f); 

	return x * x * (3.0f - 2.0f * x);
}

__device__ __forceinline__ float clamp(const float x, const float a, const float b)
{
	return fminf(fmaxf(x, a), b);
}

