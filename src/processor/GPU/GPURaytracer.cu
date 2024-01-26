#include "GPURaytracer.hpp"

void GPURaytracer::raytrace(std::shared_ptr<Scene> scene, std::shared_ptr<Camera> camera, cudaSurfaceObject_t canvas)
{
	// Camera Parameters
	CameraParams params = {};
	glm::vec4 position = glm::vec4(camera->getPosition(), 1.0f);
	glm::quat rotation = camera->getRotation();
	float exposure = camera->getExposure();
	int width = camera->getWidth();
	int height = camera->getHeight();
	cudaTextureObject_t rays = camera->getCudaRays();

	vec4transfer(params.origin, position);
	vec4transfer(params.rotation, rotation);
	params.exposure = exposure;
	params.width = width;
	params.height = height;
	params.rays = rays;

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
		data.material = Material::getGPUMaterial(obj->getMaterialName());
		objectDataArray[i++] = data;
	}
	ObjectData* objectDataArrayDev;
	checkCudaErrors(cudaMalloc((void**)&objectDataArrayDev, sizeof(ObjectData) * objects.size()));
	checkCudaErrors(cudaMemcpy(objectDataArrayDev, objectDataArray, sizeof(ObjectData) * objects.size(), cudaMemcpyHostToDevice));

	ObjectDataVector objectDataVector = {};
	objectDataVector.data = objectDataArrayDev;
	objectDataVector.size = objects.size();

	// Create Debug Output
	DebugInfo* debugInfo = nullptr;
	if (debug)
	{
		checkCudaErrors(cudaMalloc((void**)&debugInfo, sizeof(DebugInfo)));
		checkCudaErrors(cudaMemset(debugInfo, -1, sizeof(DebugInfo)));
	}

	// Launch Kernel
	int aaSamples = (bool) antiAliasing ? MAXIMUM_AA : 1;
	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, aaSamples);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

	raytraceKernel<<<gridSize, blockSize>>>(params, canvas, objectDataVector, bounceCount, maxDistance, aoIntensity, frameCount, progressiveFrameCount, debug, debugInfo);
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
	
	frameCount = frameCount + 1;
	if (progressiveRendering)
		progressiveFrameCount++;
	else 
		progressiveFrameCount = 0;
	

	checkCudaErrors(cudaFree(objectDataArrayDev));
	if (debug) checkCudaErrors(cudaFree(debugInfo));
	delete[] objectDataArray;
}



__global__ void raytraceKernel(CameraParams camera, cudaSurfaceObject_t canvas, ObjectDataVector objectDataVector, const int bounceCount, const float maxDistance, const float aoIntensity, const int frameCount, const unsigned int progressiveFrameCount, const bool debug, DebugInfo* debugInfo)
{
	__shared__ float4 sharedMemory[BLOCK_SIZE][BLOCK_SIZE][MAXIMUM_AA];
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= camera.width || y >= camera.height)
		return;

	bool isCenter = x == camera.width / 2 && y == camera.height / 2;

	unsigned int seed = (x + y * camera.width + threadIdx.z + frameCount * 719393);

	GPURay ray = setupRay(camera, x, y, bounceCount, maxDistance, seed); 

	float4 color = trace(ray, objectDataVector, aoIntensity, seed, (debug && isCenter), debugInfo);

	color = exposureCorrection(color, camera.exposure);

	sharedMemory[threadIdx.x][threadIdx.y][threadIdx.z] = color;
	__syncthreads();

	if (threadIdx.z == 0) {
		color = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
		for (int i = 0; i < blockDim.z; i++) {
			color.x += sharedMemory[threadIdx.x][threadIdx.y][i].x;
			color.y += sharedMemory[threadIdx.x][threadIdx.y][i].y;
			color.z += sharedMemory[threadIdx.x][threadIdx.y][i].z;
		}
		color.x /= blockDim.z;
		color.y /= blockDim.z;
		color.z /= blockDim.z;
	} else {
		return;
	}

	if (progressiveFrameCount != 0) {
		float4 prevColor = surf2Dread<float4>(canvas, x * sizeof(float4), y);
		color.x = (color.x + prevColor.x * progressiveFrameCount) / (progressiveFrameCount + 1);
		color.y = (color.y + prevColor.y * progressiveFrameCount) / (progressiveFrameCount + 1);
		color.z = (color.z + prevColor.z * progressiveFrameCount) / (progressiveFrameCount + 1);
	} 
	surf2Dwrite(color, canvas, x * sizeof(float4), y);
}

__device__ GPURay setupRay(const CameraParams& camera, const int x, const int y, const int bounceCount, const float maxDistance, unsigned int& seed)
{
	GPURay ray = {};
	ray.origin = camera.origin; 
	ray.bounceCount = bounceCount; 
	ray.maxDistance = maxDistance; 
	if (blockDim.z == MAXIMUM_AA) 	
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

__device__ float4 trace(GPURay& ray, const ObjectDataVector& objectDataVector, const float aoIntensity, unsigned int& seed, const bool debug, DebugInfo* debugInfo)
{
	float4 incomingLight = make_float4(0.0f, 0.0f, 0.0f, 1.0f); 
	float4 rayColor = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
	
	for (int i = 0; i < ray.bounceCount; i++)
	{
		GPURayHit closestHit = getIntersectionPoint(ray, objectDataVector); 
		if (!closestHit.didHit) break;
		GPUMaterialPositionData materialData = getMaterialData(closestHit); 

		// Debug
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
		
		if (i == 0) // First Hit
		{
			// Ambient Occlusion
			incomingLight.x += materialData.ao.x * materialData.albedo.x * aoIntensity;
			incomingLight.y += materialData.ao.y * materialData.albedo.y * aoIntensity;
			incomingLight.z += materialData.ao.z * materialData.albedo.z * aoIntensity;
		}

		// Emission
		float4 emittedLight = materialData.emission;
		incomingLight.x += emittedLight.x * emittedLight.w * rayColor.x;
		incomingLight.y += emittedLight.y * emittedLight.w * rayColor.y; 
		incomingLight.z += emittedLight.z * emittedLight.w * rayColor.z; 

		// Accumulate Color data
		rayColor.x *= materialData.albedo.x;
		rayColor.y *= materialData.albedo.y; 
		rayColor.z *= materialData.albedo.z;

		if (i < ray.bounceCount - 1) // Calculate bounce ray
		{
			ray.origin = closestHit.hitPosition;
			float4 diffuseDirection = randomUnitVectorInCosineHemisphere(seed, materialData.normal);  
			float4 specularDirection = reflect(ray.direction, materialData.normal);
			ray.direction = normalize(lerp(specularDirection, diffuseDirection, materialData.roughness));  
		}
	} 

	return incomingLight;
}

__device__ GPURayHit getIntersectionPoint(const GPURay& ray, const ObjectDataVector& dataVector)
{
	GPURayHit closestHit = {};
	closestHit.distance = FLT_MAX;
	for (int i = 0; i < dataVector.size; i++)
	{
		ObjectData data = dataVector.data[i];
		if (!intersectsBoundingBox(ray, data.mesh->minBounds, data.mesh->maxBounds)) continue;
		GPURayHit hp = getIntersectionPoint(ray, data);
		if (hp.didHit && hp.distance < closestHit.distance)
		{
			closestHit = hp;
			closestHit.objectDataIndex = i;
		}
	}
	return closestHit;
}

__device__ bool intersectsBoundingBox(const GPURay& ray, const float3& minBound, const float3& maxBound)
{
	return true;
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
	float4 normal = normalize(cross(edge2, edge1));  

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

	if (t < 0.00001 || t > ray.maxDistance)
		return output;

	
	output.distance = t;
	output.barycentricCoords.x = b0;
	output.barycentricCoords.y = b1;
	output.barycentricCoords.z = b2;
	
	return output;
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
	float theta = 2 * 3.1415926 * randomValue(seed);
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
    // output = dotProduct > 0.0 ? negate(output) : output;  // I don't know why this is necessary 
	return output;
}

