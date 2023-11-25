#pragma once

#include <iostream>
#include <unordered_map>
#include <functional>

#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h> 

#include "Shader.hpp"

typedef const std::function<void(int key, int scancode, int action, int mods)> KeyCallback;
typedef const std::function<void(double xpos, double ypos)> MouseMoveCallback;

class Window
{
public:
	Window(int width, int height, const char* name);

    int addKeyCallback(const KeyCallback callback);

    int activateKeyCallback(int id);

    int deactivateKeyCallback(int id);

    int addMouseMoveCallback(const MouseMoveCallback callback);

    int activateMouseMoveCallback(int id);

    int deactivateMouseMoveCallback(int id);

	void renderLoop();
private:
	GLFWwindow* glfwWindow;

    std::vector<std::pair<bool, KeyCallback>> keyCallbacks;
    std::vector<std::pair<bool, MouseMoveCallback>> mouseMoveCallbacks;

    void keyCallback(int key, int scancode, int action, int mods);
    void mouseMoveCallback(double xpos, double ypos);
    void resizeCallback(int width, int height);

    Shader* screenShader;
    unsigned int quadVAO{ 0 };
    unsigned int quadTexture{ 0 };

    int width, height;
    std::vector<unsigned char> data;

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

    // Camera* currentCamera;
    // std::unordered_map<const char*, Camera*> cameraMap;

    // std::unordered_map<const char*, DisplayObject*> objectMap;

    // GUI State
    bool show_demo_window = false;
    ImVec4 clear_color = ImVec4(0.5f, 0.5f, 0.5f, 1.00f);

    // GUI functions
    void displayGUI(ImGuiIO& io);
    
};