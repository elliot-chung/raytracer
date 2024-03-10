#pragma once

#include <cuda_runtime.h>

#define mat4transfer(a, b) \
	vec4transfer(a.c0, b[0]); \
	vec4transfer(a.c1, b[1]); \
	vec4transfer(a.c2, b[2]); \
	vec4transfer(a.c3, b[3]);

#define vec3transfer(a, b) \
	a.x = b.x; \
	a.y = b.y; \
	a.z = b.z;

#define vec4transfer(a, b) \
	a.x = b.x; \
	a.y = b.y; \
	a.z = b.z; \
	a.w = b.w;

struct mat3
{
	float3 c0;
	float3 c1;
	float3 c2;
};

struct mat4
{
	float4 c0;
	float4 c1;
	float4 c2;
	float4 c3;
};
