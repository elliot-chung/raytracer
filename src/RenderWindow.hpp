#pragma once

#include <iostream>
#include <unordered_map>
#include <functional>
#include <memory>
#include <chrono>
#include <iomanip>
#include <sstream>


#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h> 

#include <cuda_gl_interop.h>

#include "Shader.hpp"
#include "Camera.hpp"
#include "Scene.hpp"
#include "CustomModel.hpp"
#include "Sphere.hpp"
#include "helper_cuda.h"
#include "processor/CPU/CPURaytracer.hpp"
#include "processor/GPU/GPURaytracer.hpp"
#include "lodepng.h"

#include "global.hpp"


#define CAMERA_START_POS glm::vec3(0.0f, 0.0f, 10.0f)
#define CAMERA_START_FOV 45.0f
#define CAMERA_START_EXPOSURE 1.0f

#define CUDA_DEVICE_COUNT 16

typedef const std::function<void(int key, int scancode, int action, int mods)> KeyCallback;
typedef const std::function<void(double xpos, double ypos)> MouseMoveCallback;
typedef const std::function<void(int button, int action, int mods)> MouseButtonCallback;

class RenderWindow
{
public:
	RenderWindow(int width, int height, const char* name);

    ~RenderWindow();

    int addKeyCallback(const KeyCallback callback);

    int activateKeyCallback(int id);

    int deactivateKeyCallback(int id);

    int addMouseMoveCallback(const MouseMoveCallback callback);

    int activateMouseMoveCallback(int id);

    int deactivateMouseMoveCallback(int id);

    void clearColorData();

	void renderLoop();
private:
	GLFWwindow* glfwWindow;

    std::vector<std::pair<bool, KeyCallback>> keyCallbacks;
    std::vector<std::pair<bool, MouseMoveCallback>> mouseMoveCallbacks;
    std::vector<std::pair<bool, MouseButtonCallback>> mouseButtonCallbacks;

    void keyCallback(int key, int scancode, int action, int mods);
    void mouseMoveCallback(double xpos, double ypos);
    void mouseButtonCallback(int button, int action, int mods);
    void resizeCallback(int width, int height);

    std::unique_ptr<Shader> screenShader;
    std::unique_ptr<CPURaytracer> rtCPU;
    std::unique_ptr<GPURaytracer> rtGPU;
    std::unique_ptr<Material> defaultMaterial;
    std::shared_ptr<Camera> camera;
    std::shared_ptr<Scene> scene;

    unsigned int quadVAO{ 0 };
    unsigned int quadTexture{ 0 };

    cudaSurfaceObject_t bitmap_surface = 0;
    cudaGraphicsResource* resource = 0;

    int width, height;
    bool vsync = false;
    std::vector<float> data;

    float quadVertices[24] = {
        // positions   // texCoords
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };

    
    bool useGPU = false;

    void updateTexture();

    void displayWindowGUI(ImGuiIO& io);

    void createSurfaceObject();

    void takeScreenshot();
    
};