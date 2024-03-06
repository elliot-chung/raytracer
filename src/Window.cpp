#include "Window.hpp"

bool usingGPU;
bool availableGPU;

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
            availableGPU = false;
            std::cout << "No CUDA devices found" << std::endl;
        }
        else
        {
            availableGPU = true;
            std::cout << "Found " << numDevices << " CUDA devices" << std::endl;
        }
        useGPU = availableGPU;
    }

    // Heap allocate objects
    {
        screenShader = std::make_unique<Shader>("C:/Users/ec201/OneDrive/Desktop/raytracer/src/shaders/vert.shader", "C:/Users/ec201/OneDrive/Desktop/raytracer/src/shaders/frag.shader");
        camera = std::make_shared<Camera>(CAMERA_START_POS, glm::quat(), CAMERA_START_FOV, CAMERA_START_EXPOSURE);
        scene = std::make_shared<Scene>();

        rtCPU = std::make_unique<CPURaytracer>();
        rtGPU = std::make_unique<GPURaytracer>();

        screenShader->use();
        screenShader->setInt("screenTexture", 0);

        camera->setHeight(height);
        camera->setWidth(width);
    }
}

inline void Window::updateTexture()
{
    glBindTexture(GL_TEXTURE_2D, quadTexture);
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
void Window::createSurfaceObject()
{
    if (bitmap_surface) checkCudaErrors(cudaDestroySurfaceObject(bitmap_surface));

    checkCudaErrors(cudaGraphicsMapResources(1, &resource, 0));

    cudaArray* internalArray;

    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&internalArray, resource, 0, 0));

    cudaResourceDesc resDesc;

    memset(&resDesc, 0, sizeof(resDesc));

    resDesc.resType = cudaResourceTypeArray;

    resDesc.res.array.array = internalArray;

    checkCudaErrors(cudaCreateSurfaceObject(&bitmap_surface, &resDesc));
}

void Window::renderLoop()
{
    Material mat1("floormat", glm::vec4(1.0f, 1.0f, 1.0f, 1.0f), 1.0f);

    Material mat2("lightmat", glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), 1.0f, 0.0f, glm::vec3(1.0f, 1.0f, 1.0f), 1.0f); 

    Material mat3("smoothhalfmetal", glm::vec4(1.0f, 0.0f, 0.0f, 1.0f), 0.0f, 0.5f);
    Material mat4("halfsmoothmetal", glm::vec4(1.0f, 0.0f, 0.0f, 1.0f), 0.5f, 1.0f);
    Material mat5("roughhalfmetal", glm::vec4(1.0f, 0.0f, 0.0f, 1.0f), 1.0f, 0.5f);
    Material mat6("halfsmoothdielectric", glm::vec4(1.0f, 0.0f, 0.0f, 1.0f), 0.5f, 0.0f);
    Material mat7("pbrmetal");
    Material mat8("pbrbrick");

    mat7.setAlbedoTexture("res/metalmaterial/Metal046B_1K-PNG_Color.png");
    mat7.setNormalTexture("res/metalmaterial/Metal046B_1K-PNG_NormalGL.png");
    mat7.setMetalTexture("res/metalmaterial/Metal046B_1K-PNG_Metalness.png");
    mat7.setRoughnessTexture("res/metalmaterial/Metal046B_1K-PNG_Roughness.png");

    mat8.setAlbedoTexture("res/modern-brick1_bl/modern-brick1_albedo.png");
    mat8.setNormalTexture("res/modern-brick1_bl/modern-brick1_normal-ogl.png");
    mat8.setMetalTexture("res/modern-brick1_bl/modern-brick1_metallic.png");
    mat8.setRoughnessTexture("res/modern-brick1_bl/modern-brick1_roughness.png");
    mat8.setAmbientOcclusionTexture("res/modern-brick1_bl/modern-brick1_ambient-occlusion.png");

    

    mat1.sendToGPU();
    mat2.sendToGPU();
    mat3.sendToGPU();
    mat4.sendToGPU();
    mat5.sendToGPU();
    mat6.sendToGPU();
    mat7.sendToGPU();
    mat8.sendToGPU();
   

    Cube cube(glm::vec3(0.0f, -1.0f, 0.0f), glm::quat(), glm::vec3(100.0f, 1.0f, 100.0f));
    scene->addToScene("cube", &cube);
    cube.setMaterialName("floormat");

    Cube cube2(glm::vec3(0.0f, 2.0f, 0.0f));
    scene->addToScene("cube2", &cube2);
    cube2.setMaterialName("lightmat");

    Cube cube3(glm::vec3(-1.5f, 0.0f, -1.5f));  
    scene->addToScene("cube3", &cube3); 
    cube3.setMaterialName("smoothhalfmetal"); 

    Cube cube4(glm::vec3(1.5f, 0.0f, -1.5f));
    scene->addToScene("cube4", &cube4);
    cube4.setMaterialName("halfsmoothmetal");

    Cube cube5(glm::vec3(-1.5f, 0.0f, 1.5f));
	scene->addToScene("cube5", &cube5);
    cube5.setMaterialName("roughhalfmetal");

    Cube cube6(glm::vec3(1.5f, 0.0f, 1.5f));
    scene->addToScene("cube6", &cube6);
    cube6.setMaterialName("halfsmoothdielectric");

    Sphere sphere(glm::vec3(0.0f, 4.0f, 0.0f)); 
    scene->addToScene("sphere", &sphere); 
    sphere.setMaterialName("pbrbrick");
    ImGuiIO& io = ImGui::GetIO();

    while (!glfwWindowShouldClose(glfwWindow))
    {
        glfwPollEvents();

        
        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Toggle cuda surface object
        if (useGPU && !usingGPU)
        {
            checkCudaErrors(cudaGraphicsGLRegisterImage(&resource, quadTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));
            createSurfaceObject();
            usingGPU = true;
        }
		else if (!useGPU && usingGPU)
		{
			checkCudaErrors(cudaGraphicsUnmapResources(1, &resource, 0));
			checkCudaErrors(cudaGraphicsUnregisterResource(resource));
			usingGPU = false;
		}

        // GUI
        displayWindowGUI(io);
        camera->updateGUI(io);  
        scene->updateGUI(io);

        // Object updates
        camera->update(io);
        scene->update(io);
        
        // Raytrace
        if (usingGPU)
        {
            rtGPU->raytrace(scene, camera, bitmap_surface);
        }
        else
        {
            data = rtCPU->raytrace(scene, camera);
            updateTexture();
        }
            
        
        // Render texture to screen
        glBindVertexArray(quadVAO);
        glBindTexture(GL_TEXTURE_2D, quadTexture);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        
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

inline void flipVertical(unsigned char* image, unsigned int width, unsigned int height, unsigned int channels)
{
    unsigned char* temp = new unsigned char[width * channels];
    for (unsigned int i = 0; i < height / 2; i++)
    {
        memcpy(temp, &image[i * width * channels], width * channels);
        memcpy(&image[i * width * channels], &image[(height - i - 1) * width * channels], width * channels);
        memcpy(&image[(height - i - 1) * width * channels], temp, width * channels);
    }
}

void Window::displayWindowGUI(ImGuiIO& io)
{
	static bool show_demo_window = false;    
    static int counter = 1;  // Eventually replace with a timestamp
    static bool progressiveRendering = rtCPU->getProgressiveRendering();
    static bool antiAliasing = (bool)rtCPU->getAntiAliasingEnabled();
    
    static int bounceCount = rtCPU->getBounceCount();
    static float maxDistance = rtCPU->getMaxDistance();
    static float aoIntensity = rtCPU->getAOIntensity(); 

    static int sampleRate = rtCPU->getSampleCount();

    static float lightYaw = 0.0f;
    static float lightPitch = 0.0f;
    static float4 lightColor = make_float4(1.0f, 1.0f, 1.0f, 100.0f);
    static float4 skyColor = make_float4(0.5f, 0.5f, 1.0f, 0.3f);

    if (show_demo_window) ImGui::ShowDemoWindow(&show_demo_window);

    ImGui::Begin("Render Options");                          

    ImGui::Text("GPU: %s", availableGPU ? "Available" : "Not Available");
    ImGui::Checkbox("Use GPU", &useGPU);      
    ImGui::SameLine();  ImGui::Checkbox("Demo Window", &show_demo_window);    
    ImGui::SameLine();  ImGui::Checkbox("Progressive Rendering", &progressiveRendering);
    ImGui::SameLine();  ImGui::Checkbox("Anti-Aliasing", &antiAliasing);

    ImGui::InputInt("Bounce Count", &bounceCount, 1, 2);
    ImGui::InputFloat("Max Distance", &maxDistance, 1.0f, 10.0f);
    ImGui::InputFloat("AO Intensity", &aoIntensity, 0.005f, 0.05f);
    ImGui::InputInt("Sample Rate", &sampleRate, 1, 2);
    ImGui::InputFloat("Light Yaw", &lightYaw, 0.1f, 0.2f);
    ImGui::InputFloat("Light Pitch", &lightPitch, 0.1f, 0.2f);
    ImGui::ColorEdit3("Light Color", &lightColor.x);
    ImGui::InputFloat("Light Intensity", &lightColor.w, 10.0f, 100.0f);
    ImGui::ColorEdit3("Sky Color", &skyColor.x); 
    ImGui::InputFloat("Sky Intensity", &skyColor.w, 10.0f, 100.0f);

	{
        rtGPU->setBounceCount(bounceCount);
        rtGPU->setMaxDistance(maxDistance);
        rtGPU->setAOIntensity(aoIntensity);
        rtGPU->setProgressiveRendering(progressiveRendering);
        rtGPU->setAntiAliasingEnabled(antiAliasing);
        rtGPU->setSampleCount(sampleRate);
        rtGPU->setSkyLight(lightPitch, lightYaw, lightColor, skyColor);

        rtGPU->setDebug(ImGui::Button("Debug"));

		rtCPU->setBounceCount(bounceCount);
		rtCPU->setMaxDistance(maxDistance);
		rtCPU->setAOIntensity(aoIntensity);
        rtCPU->setProgressiveRendering(progressiveRendering);
        rtCPU->setSampleCount(sampleRate);
        rtCPU->setAntiAliasingEnabled(antiAliasing);
        rtGPU->setSkyLight(lightPitch, lightYaw, lightColor, skyColor); 
	}

    if (ImGui::Button("Take Screenshot"))
    {
        unsigned char* pixels = new unsigned char[width * height * 4];
        glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

        flipVertical(pixels, width, height, 4);

        std::string path = "screenshot" + std::to_string(counter++) + ".png";
        unsigned int error = lodepng::encode(path, pixels, width, height);

        delete[] pixels;
    }

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

    if (usingGPU)
    {
        checkCudaErrors(cudaGraphicsUnmapResources(1, &resource, 0));
        checkCudaErrors(cudaGraphicsUnregisterResource(resource));
        updateTexture();
        checkCudaErrors(cudaGraphicsGLRegisterImage(&resource, quadTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
        createSurfaceObject();
    }
    

    {
        camera->setHeight(height);
        camera->setWidth(width);
    }
}