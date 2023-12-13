#include "Window.hpp"

Window::Window(int width, int height, const char* name)
{
    this->width = width;
    this->height = height;

    // Setup OpenGL
    {
        // Initialize GLFW
        if (!glfwInit())
            throw std::runtime_error("Failed to initialize GLFW");

        // OpenGL 3.3 Core Profile
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Mac

        // Create glfwWindow
        glfwWindow = glfwCreateWindow(width, height, name, nullptr, nullptr);
        if (glfwWindow == nullptr)
        {
            glfwTerminate();
            throw std::runtime_error("Failed to create GLFW window");
        }
        glfwMakeContextCurrent(glfwWindow);

        // Initialize GLAD
        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
        {
            glfwTerminate();
            throw std::runtime_error("Failed to initialize GLAD");
        }

        // Set Callbacks
        glfwSetWindowUserPointer(glfwWindow, this); // Set User Pointer

        addKeyCallback([this](int key, int scancode, int action, int mods)
            {
                if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
                    glfwSetWindowShouldClose(glfwWindow, true);
            });

        glfwSetKeyCallback(glfwWindow, [](GLFWwindow* window, int key, int scancode, int action, int mods)
            {
                Window* win = (Window*)glfwGetWindowUserPointer(window);
                win->keyCallback(key, scancode, action, mods);
            });

        glfwSetCursorPosCallback(glfwWindow, [](GLFWwindow* window, double xpos, double ypos)
            {
                Window* win = (Window*)glfwGetWindowUserPointer(window);
                win->mouseMoveCallback(xpos, ypos);
            });

        glfwSetFramebufferSizeCallback(glfwWindow, [](GLFWwindow* window, int width, int height)
            {
                Window* win = (Window*)glfwGetWindowUserPointer(window);
                win->resizeCallback(width, height);
            }); // Set Resize Callback
    }

    // Setup Dear ImGui
    {
        const char* glsl_version = "#version 330";
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;         // Enable Docking
        io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;       // Enable Multi-Viewport / Platform Windows
        //io.ConfigViewportsNoAutoMerge = true;
        //io.ConfigViewportsNoTaskBarIcon = true;

        // Setup Dear ImGui style
        ImGui::StyleColorsDark();
        //ImGui::StyleColorsLight();

        // When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones.
        ImGuiStyle& style = ImGui::GetStyle();
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            style.WindowRounding = 0.0f;
            style.Colors[ImGuiCol_WindowBg].w = 1.0f;
        }

        // Setup Platform/Renderer backends
        ImGui_ImplGlfw_InitForOpenGL(glfwWindow, true);
        ImGui_ImplOpenGL3_Init(glsl_version);
    }

    // Setup fullscreen quad in OpenGL
    {
        unsigned int quadVBO;
        glGenVertexArrays(1, &quadVAO);
        glGenBuffers(1, &quadVBO);
        glBindVertexArray(quadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

        glEnable(GL_FRAMEBUFFER_SRGB); // Gamma Correction

        // generate texture  
        glGenTextures(1, &quadTexture);
        glBindTexture(GL_TEXTURE_2D, quadTexture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        // glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        // Initialize texture Data
        clearColorData();
        updateTexture();
    }

    // Check for CUDA devices
    {
        unsigned int numDevices = CUDA_DEVICE_COUNT;
        int devices[CUDA_DEVICE_COUNT];

        cudaGLGetDevices(&numDevices, devices, numDevices, cudaGLDeviceListAll);
        if (numDevices == 0)
        {
            useGPU = false;
            std::cout << "No CUDA devices found" << std::endl;
        }
        else
        {
            useGPU = true;
            std::cout << "Found " << numDevices << " CUDA devices" << std::endl;
        }
    }

    // Setup CUDA
    if (useGPU) 
    {
        cudaArray* internalArray;
        cudaGraphicsResource* resource;

        cudaGraphicsGLRegisterImage(&resource, quadTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
  
        cudaGraphicsMapResources(1, &resource, 0);

        cudaGraphicsSubResourceGetMappedArray(&internalArray, resource, 0, 0);
        
        cudaResourceDesc resDesc;

        memset(&resDesc, 0, sizeof(resDesc));
        
        resDesc.resType = cudaResourceTypeArray;

        resDesc.res.array.array = internalArray;

        cudaCreateSurfaceObject(&bitmap_surface, &resDesc);
    }

    // Heap allocate objects
    {
        screenShader = std::make_unique<Shader>("src/shaders/vert.shader", "src/shaders/frag.shader");
        camera = std::make_shared<Camera>(CAMERA_START_POS, glm::quat(), CAMERA_START_FOV, CAMERA_START_EXPOSURE, useGPU);
        scene = std::make_shared<Scene>();

        rtCPU = std::make_unique<CPURaytracer>();
        rtGPU = std::make_unique<GPURaytracer>();

        screenShader->use();
        screenShader->setInt("screenTexture", 0);

        camera->calcRays(width, height);
    }
}

inline void Window::updateTexture()
{
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, &data[0]);
}
int Window::addKeyCallback(const KeyCallback callback)
{
    keyCallbacks.push_back(std::make_pair(true, callback));
    return keyCallbacks.size() - 1;
}
int Window::activateKeyCallback(int id)
{
    if (id >= keyCallbacks.size()) return -1;
    keyCallbacks[id].first = true;
    return id;
}
int Window::deactivateKeyCallback(int id)
{
    if (id >= keyCallbacks.size()) return -1;
    keyCallbacks[id].first = false;
    return id;
}
int Window::addMouseMoveCallback(const MouseMoveCallback callback)
{
    mouseMoveCallbacks.push_back(std::make_pair(true, callback));
    return keyCallbacks.size() - 1;
}
int Window::activateMouseMoveCallback(int id)
{
    if (id >= mouseMoveCallbacks.size()) return -1;
    mouseMoveCallbacks[id].first = true;
    return id;
}
int Window::deactivateMouseMoveCallback(int id)
{
    if (id >= mouseMoveCallbacks.size()) return -1;
    mouseMoveCallbacks[id].first = false;
    return id;
}
void Window::clearColorData()
{
    data = std::vector<float>(width * height * 4, 0.5f);
}

void Window::renderLoop()
{
    // Setup Scene (this is temporary)
    rtCPU->setMaxDistance(100.0f);
    rtCPU->setBounceCount(3);

    Material mat1("redmat", glm::vec4(1.0f, 0.0f, 0.0f, 1.0f), 1.0f);
    Material mat2("bluemat", glm::vec4(0.0f, 0.0f, 1.0f, 1.0f), 1.0f);
    Material mat3("lightmat", glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), 1.0f, 0.0f, glm::vec3(1.0f, 1.0f, 1.0f), 10.0f);

    Cube cube(glm::vec3(0.0f, -1.0f, 0.0f), glm::quat(), glm::vec3(100.0f, 1.0f, 100.0f));
    scene->addToScene("cube", &cube);
    cube.setMaterialName("redmat");

    Cube cube2(glm::vec3(0.0f, 0.2f, 0.0f));
    scene->addToScene("cube2", &cube2);
    cube2.setMaterialName("bluemat");
    
    Cube cube3(glm::vec3(0.0f, 2.0f, 0.0f));
    scene->addToScene("cube3", &cube3);
    cube3.setMaterialName("lightmat");

    
    ImGuiIO& io = ImGui::GetIO();

    while (!glfwWindowShouldClose(glfwWindow))
    {
        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wa  nts to use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
        glfwPollEvents();

        
        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        


        displayWindowGUI(io);
        camera->updateGUI(io);  
        scene->updateGUI(io);


        camera->update(io);
        scene->update(io);
        
        if (useGPU)
        {
            rtGPU->trace(scene, camera, bitmap_surface);
        }
        else
        {
            data = rtCPU->trace(scene, camera);
            updateTexture();
        }
            
        
        //----------------------------

        glBindVertexArray(quadVAO);
        glBindTexture(GL_TEXTURE_2D, quadTexture);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        
        // Rendering
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());


        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            GLFWwindow* backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }
        

        glfwSwapBuffers(glfwWindow);
    }
}

Window::~Window()
{
    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(glfwWindow);
    glfwTerminate();
}

void Window::displayWindowGUI(ImGuiIO& io)
{
	static bool show_demo_window = false;    
    static float f = 0.0f;
    static int counter = 0;

    if (show_demo_window) ImGui::ShowDemoWindow(&show_demo_window);

    ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

    ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
    ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state

    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
    ImGui::End();
}

void Window::keyCallback(int key, int scancode, int action, int mods)
{
    for (auto callback : keyCallbacks)
        if (callback.first)
            callback.second(key, scancode, action, mods);
}

void Window::mouseMoveCallback(double xpos, double ypos)
{
    for (auto callback : mouseMoveCallbacks)
        if (callback.first)
            callback.second(xpos, ypos);
}

void Window::resizeCallback(int width, int height)
{
    glViewport(0, 0, width, height);
    this->width = width;
    this->height = height;
    clearColorData();
    camera->calcRays(width, height);
}