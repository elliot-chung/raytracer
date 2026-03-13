# CUDA Raytracer Project

A C++/CUDA path tracer with an OpenGL display front-end, built to demonstrate core computer graphics concepts. The app loads 3D models (via Assimp), renders them with a physically based path tracer on either the CPU or GPU, and allows interactive camera control, object placement, and material editing through a real-time GUI.

## Example Renders
![Sword Render 1](20240610102036.png)

![Sword Render 2](20240610102209.png)

---

## Architecture Overview

- **Host (C++)**: GLFW window, ImGui UI, scene graph, camera, and model loading. The host builds scene and camera data and either invokes the **CPU raytracer** or the **CUDA raytracer**, then displays the result.
- **Display**: A fullscreen quad is drawn in OpenGL; when using the GPU, the raytracer writes directly into an OpenGL texture via **CUDA–OpenGL interop** (`cudaGraphicsGLRegisterImage` / surface objects), so no CPU readback is needed for display.
- **Dual backends**: The same scene, camera, and renderer settings (bounces, samples, AO, skylight, etc.) drive both a single-threaded CPU path tracer and a CUDA kernel. You can switch between them at runtime from the Options window (e.g. for debugging or when no CUDA device is available).

---

## Graphics Concepts Implemented

### Path tracing and sampling
- **Path tracing** with configurable bounce count and max ray distance.
- **Per-pixel multi-sampling** with optional jitter (anti-aliasing); samples are accumulated and averaged.
- **Progressive refinement**: in GPU mode, frames can be accumulated over time (running average with previous frame) for cleaner convergence.
- **Cosine-weighted hemisphere sampling** for diffuse and a roughness-based blend toward specular for the next path direction.
- **HDR exposure**: result is tone-mapped with `1 - exp(-color * exposure)` before display.

### PBR (Cook–Torrance BRDF)
- **Cook–Torrance** microfacet model:
  - **GGX** (Trowbridge–Reitz) normal distribution.
  - **Smith** geometry term (Schlick-GGX for visibility).
  - **Schlick** approximation for Fresnel; F0 blended by metalness.
- **Metalness/roughness workflow**: albedo, roughness, and metalness (with optional texture maps) drive the BRDF; emission is supported for first-hit and along the path.

### Ray–geometry
- **Ray–AABB**: ray vs axis-aligned box in object space (inverse model matrix used to transform ray into model space). Used as an early reject per object before ray–triangle tests.
- **Ray–triangle**: robust ray–triangle intersection with barycentric coordinates. Implementation uses axis permutation (largest component of the ray direction) for numerical stability and double precision for edge tests when needed (Möller–Trumbore–style).
- **Per-object composite bounds**: each `DisplayObject` has a composite AABB over all its meshes for a single cheap reject before iterating meshes/triangles.

### Materials and shading
- **TBN from UVs**: tangent and bitangent are derived from the triangle’s UV deltas; the normal is either interpolated from vertex normals or from the triangle. **Gram–Schmidt** is used to orthogonalize the tangent to the normal, then the bitangent is recomputed so the TBN matrix correctly transforms normal-map vectors to world space.
- **Texture mapping**: albedo, normal, roughness, metalness, ambient occlusion, and emission can be driven by textures. On the GPU, materials and textures are mirrored to device memory (CUDA texture objects for sampling); the host `Material` class pushes updates via `sendToGPU()`.

### Lighting and environment
- **Directional “sun”**: single directional light with configurable direction (yaw/pitch) and color/intensity, combined with a simple sky color and intensity.
- **Ambient occlusion**: AO term (from material or default) is applied on the first hit only, scaled by a global AO intensity.

---

## Project Structure (high level)

| Area | Role |
|------|------|
| `main.cpp` | Creates `RenderWindow` and runs the render loop. |
| `RenderWindow` | GLFW/OpenGL/ImGui setup, CUDA–GL interop, CPU/GPU toggle, camera/scene/material GUI, screenshot (lodepng). |
| `Scene` | Scene graph (`DisplayObject` map); builds `GPUObjectDataVector` (model matrices, inverse, and pointers to GPU mesh/material data) for the CUDA raytracer each frame. |
| `Camera` | Position, quaternion rotation, FOV, exposure. Precomputes primary ray directions (and uploads them as a CUDA texture when using GPU). |
| `DisplayObject` | Transform (position/rotation/scale), composite AABB, list of meshes + material indices. Packs mesh and material pointers into `LLGPUObjectData` and uploads to GPU. |
| `CustomModel` | Loads models via Assimp (triangulate, UVs, flip UVs); extracts materials and textures (albedo, normal, roughness, metalness, AO, emission); builds meshes and sends them to the GPU. |
| `Material` / `GPUMaterial` | Host material with optional textures; `GPUMaterial` is the device-side mirror with CUDA texture objects. Same interface for albedo, normal, roughness, metal, AO, emission. |
| `processor/GPU/GPURaytracer` | CUDA kernel: one thread per pixel (with a small 3D block for multiple samples per pixel). Ray setup from camera texture, path loop, ray–AABB then ray–triangle, material evaluation, Cook–Torrance next direction, exposure, optional progressive blend, write to surface. |
| `processor/CPU/CPURaytracer` | Single-threaded path tracer over the same scene/camera/settings; fills a float buffer that is then uploaded to the OpenGL texture. |
| `linal.hpp` | Device-side `mat3`/`mat4` (column vectors) and transfer macros from GLM for use in CUDA code. |

The GPU kernel uses **8×8×4** thread blocks: each (x,y) pixel can run up to 4 sample iterations in the z dimension; partial colors are reduced in shared memory and the final average is written by the thread with `threadIdx.z == 0`.

---

## Tech Stack

- **C++** (host), **CUDA** (device kernels and memory)
- **OpenGL 3.3** (Core), **GLFW**, **GLAD**
- **Dear ImGui** (docking, viewports)
- **CUDA–OpenGL interop** (render target is a registered GL texture)
- **GLM** (host math: vectors, matrices, quaternions)
- **Assimp** (model loading: meshes, materials, textures)
- **stb_image** (image loading for textures)
- **lodepng** (screenshot encoding)

---

## Building the Project

1. Install **Visual Studio 2022** (or compatible).
2. Clone the repository.
3. [Download](https://www.glfw.org/download) and link **GLFW**.
4. [Download](https://github.com/assimp/assimp) and link **Assimp**.
5. Ensure **CUDA Toolkit** is installed and the project is configured to build the `.cu` files with the CUDA compiler.
6. Open the solution (`.sln`) in Visual Studio and build.

At runtime, if no CUDA-capable device is found (or CUDA–GL interop is unavailable), the app reports “GPU: Not Available” and uses the CPU raytracer; you can still toggle “Use GPU” in the Options window when a device is available.
