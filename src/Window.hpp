#pragma once

#include <iostream>
#include <unordered_map>

#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h> 

#include "Shader.hpp"

class Window
{
public:
	Window(int width, int height, const char* name);

    void updateTexture();

	void renderLoop();
private:
	GLFWwindow* glfwWindow;

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

    // Camera* currentCamera;
    // std::unordered_map<const char*, Camera*> cameraMap;

    // std::unordered_map<const char*, DisplayObject*> objectMap;

    // GUI State
    bool show_demo_window = false;
    ImVec4 clear_color = ImVec4(0.5f, 0.5f, 0.5f, 1.00f);

    // GUI functions
    void displayGUI(ImGuiIO& io);
};