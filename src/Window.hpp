#pragma once

#include <iostream>
#include <unordered_map>
#include <functional>
#include <memory>

#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h> 

#include "Shader.hpp"
#include "Camera.hpp"
#include "Scene.hpp"
#include "Cube.hpp"
#include "Material.hpp"
#include "processor/CPU/CPURaytracer.hpp"

#include "glm/gtx/string_cast.hpp"

#define CAMERA_START_POS glm::vec3(0.0f, 0.0f, 2.0f)
#define CAMERA_START_FOV 45.0f

typedef const std::function<void(int key, int scancode, int action, int mods)> KeyCallback;
typedef const std::function<void(double xpos, double ypos)> MouseMoveCallback;


class Window
{
public:
	Window(int width, int height, const char* name);

    ~Window();

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

    void keyCallback(int key, int scancode, int action, int mods);
    void mouseMoveCallback(double xpos, double ypos);
    void resizeCallback(int width, int height);

    std::unique_ptr<Shader> screenShader;
    std::shared_ptr<Camera> camera;
    std::shared_ptr<Scene> scene;

    unsigned int quadVAO{ 0 };
    unsigned int quadTexture{ 0 };

    int width, height;
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

    void updateTexture();


    // GUI functions
    void displayWindowGUI(ImGuiIO& io);

    

    
};