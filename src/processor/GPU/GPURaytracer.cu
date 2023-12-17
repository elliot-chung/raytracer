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
	for (auto & objPair : objects)
	{
		DisplayObject* obj = objPair.second;
		ObjectData data = {};
		mat4transfer(data.modelMatrix, obj->getModelMatrix());
		data.mesh = obj->getMesh()->getGPUMeshData();
		objectDataArray[i++] = data;
	}
	ObjectData* objectDataArrayDev;
	checkCudaErrors(cudaMalloc((void**)&objectDataArrayDev, sizeof(ObjectData) * objects.size()));
	checkCudaErrors(cudaMemcpy(objectDataArrayDev, objectDataArray, sizeof(ObjectData) * objects.size(), cudaMemcpyHostToDevice));

	ObjectDataVector objectDataVector = {};
	objectDataVector.data = objectDataArrayDev;
	objectDataVector.size = objects.size();

	// Send Rays to GPU

	// Launch Kernel
	dim3 blockSize(16, 16);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

	raytraceKernel<<<gridSize, blockSize>>>(params, canvas, objectDataVector, bounceCount, maxDistance);
	checkCudaErrors(cudaPeekAtLastError()); 
	checkCudaErrors(cudaDeviceSynchronize()); 
	
	

	checkCudaErrors(cudaFree(objectDataArrayDev));
	delete[] objectDataArray;
}

__global__ void raytraceKernel(CameraParams camera, cudaSurfaceObject_t canvas, ObjectDataVector objectDataVector, int bounceCount, float maxDistance)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= camera.width || y >= camera.height)
		return;


	GPURay ray = setupRay(camera, x, y, bounceCount, maxDistance);

	float4 color = singleTrace(ray, objectDataVector);
	// color = exposureCorrection(color, camera.exposure);

	surf2Dwrite(color, canvas, x * sizeof(float4), y);
}

__device__ GPURay setupRay(const CameraParams& camera, const int x, const int y, const int bounceCount, const float maxDistance)
{
	GPURay ray = {};
	ray.origin = camera.origin;
	ray.direction = tex2D<float4>(camera.rays, x, y);
	ray.direction = rotate(ray.direction, camera.rotation);
	ray.bounceCount = bounceCount;
	ray.maxDistance = maxDistance;
	return ray;
}

__device__ float4 singleTrace(GPURay& ray, ObjectDataVector objectDataVector)
{
	GPURayHit closestHit = {};
	closestHit.distance = FLT_MAX;
	for (int i = 0; i < objectDataVector.size; i++)
	{
		ObjectData data = objectDataVector.data[i];
		if (!intersectsBoundingBox(ray, data.mesh->minBounds, data.mesh->maxBounds)) continue;
		GPURayHit hp = getIntersectionPoint(ray, data);
		if (hp.didHit && hp.distance < closestHit.distance)
		{
			closestHit = hp;
		}
	}

	if (closestHit.didHit) return make_float4(1.0f, 0.0f, 0.0f, 1.0f);
	else return make_float4(0.0f, 0.0f, 0.0f, 1.0f);
}

__device__ bool intersectsBoundingBox(const GPURay& ray, const float3& minBound, const float3& maxBound)
{
	return true;
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
	/*
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

	float2 uv0 = make_float2(uvCoords[i0uv + 0], uvCoords[i0uv + 1]); 
	float2 uv1 = make_float2(uvCoords[i1uv + 0], uvCoords[i1uv + 1]); 
	float2 uv2 = make_float2(uvCoords[i2uv + 0], uvCoords[i2uv + 1]); 

	float3 edge1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z); 
	float3 edge2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z); 
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
	tangent = matVecMul(modelMatrix, tangent);
	bitangent = matVecMul(modelMatrix, bitangent);
	tangent = normalize(tangent);
	bitangent = normalize(bitangent);
	float4 normal = negate(cross(tangent, bitangent));

	mat3 tbnMatrix;
	tbnMatrix.c0 = make_float3(tangent.x, tangent.x, tangent.x); 
	tbnMatrix.c1 = make_float3(bitangent.y, bitangent.y, bitangent.y); 
	tbnMatrix.c2 = make_float3(normal.z, normal.z, normal.z); 

	float4 interpPosition = make_float4(
		v0.x * barycentricCoords.x + v1.x * barycentricCoords.y + v2.x * barycentricCoords.z,
		v0.y * barycentricCoords.x + v1.y * barycentricCoords.y + v2.y * barycentricCoords.z,
		v0.z * barycentricCoords.x + v1.z * barycentricCoords.y + v2.z * barycentricCoords.z,
		1.0f
	);
	float2 interpUV = make_float2(
		uv0.x * barycentricCoords.x + uv1.x * barycentricCoords.y + uv2.x * barycentricCoords.z,
		uv0.y * barycentricCoords.x + uv1.y * barycentricCoords.y + uv2.y * barycentricCoords.z
	);

	interpPosition = matVecMul(modelMatrix, interpPosition);

	
	ray.hitInfo.hitPosition = interpPosition;
	ray.hitInfo.uv = interpUV;
	// ray.hitInfo.material = meshData->material;
	ray.hitInfo.tbnMatrix = tbnMatrix;
	*/

	output.didHit = true;

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

	
	if (e0 == 0.0f || e1 == 0.0f || e2 == 0.0f) // double precision recomputation in the unlikely case any edge coefficient is 0
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

	if (t < 0 || t > ray.maxDistance)
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

__device__ __forceinline__ float4 negate(const float4 v)
{
	return make_float4(-v.x, -v.y, -v.z, v.w);
}


__device__ __forceinline__ float4 normalize(const float4 v)
{
	float invLength = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
	return make_float4(v.x * invLength, v.y * invLength, v.z * invLength, v.w);
}




