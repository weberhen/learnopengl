# Henrique Weber, 2018
# adapted from https://learnopengl.com/Lighting/Colors
# This file is licensed under the MIT License.

import glfw
from OpenGL.GL import *
import numpy as np
import glm
import Imath
import OpenEXR
from learnopengl.ibl.shader import Shader
from learnopengl.ibl.render_cube import renderCube
from learnopengl.ibl.camera import Camera


def load_hdr(path):
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    rgb_img_openexr = OpenEXR.InputFile(path)
    rgb_img = rgb_img_openexr.header()['dataWindow']
    size_img = (rgb_img.max.x - rgb_img.min.x + 1, rgb_img.max.y - rgb_img.min.y + 1)

    redstr = rgb_img_openexr.channel('R', pt)
    red = np.fromstring(redstr, dtype=np.float32)
    red.shape = (size_img[1], size_img[0])

    greenstr = rgb_img_openexr.channel('G', pt)
    green = np.fromstring(greenstr, dtype=np.float32)
    green.shape = (size_img[1], size_img[0])

    bluestr = rgb_img_openexr.channel('B', pt)
    blue = np.fromstring(bluestr, dtype=np.float32)
    blue.shape = (size_img[1], size_img[0])

    hdr_img = np.dstack((red, green, blue))

    return hdr_img


def framebuffer_size_callback(window, width, height):
    glViewport(0, 0, width, height)


camera_pos = glm.vec3(0.0, 0.0, 3.0)
camera_front = glm.vec3(0.0, 0.0, -1.0)
camera_up = glm.vec3(0.0, 1.0, 0.0)
light_pos = glm.vec3(1.2, 1.0, 2.0)

delta_time = 0.0
last_frame = 0.0


def process_input(window):
    if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
        glfw.set_window_should_close(window, True)

    global camera_pos, camera_front, camera_up, delta_time
    camera_speed = 2.5 * delta_time
    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
        camera_pos += camera_speed * camera_front
    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
        camera_pos -= camera_speed * camera_front
    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
        camera_pos -= glm.normalize(glm.cross(camera_front, camera_up)) * camera_speed
    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
        camera_pos += glm.normalize(glm.cross(camera_front, camera_up)) * camera_speed
    if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
        camera_pos -= camera_speed * glm.vec3(0.0, -1.0, 0.0)
    if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
        camera_pos += camera_speed * glm.vec3(0.0, -1.0, 0.0)


def main():
    global current_frame, last_frame, delta_time, light_pos
    # initialize glfw
    if not glfw.init():
        return

    screen_width = 800
    screen_height = 600
    window = glfw.create_window(screen_width, screen_height, "LearnOpenGL", None, None)

    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    camera = Camera(glm.vec3(0.0, 0.0, 0.3))

    # configure global opengl state
    # -----------------------------
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)  # set depth function to less than AND equal for skybox depth trick.

    envmap = load_hdr(
        '/gel/usr/heweb4/Documents/_datasets/envmaps_reexposed_rotated/test/output-03-18-9C4A7695-9C4A7695_Panorama_hdr_inpainted-result_000.exr')
    hdrTexture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, hdrTexture)
    width, height = envmap.shape[0:2]
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, width, height, 0, GL_RGB, GL_FLOAT, envmap)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    cubemap_vertex_shader = """
    #version 330 core
    layout (location = 0) in vec3 aPos;
    
    out vec3 localPos;
    
    uniform mat4 projection;
    uniform mat4 view;
    
    void main()
    {
        localPos = aPos;  
        gl_Position =  projection * view * vec4(localPos, 1.0);
    }
    """

    cubemap_fragment_shader = """
    #version 330 core
    out vec4 FragColor;
    in vec3 localPos;
    
    uniform sampler2D equirectangularMap;
    
    const vec2 invAtan = vec2(0.1591, 0.3183);
    vec2 SampleSphericalMap(vec3 v)
    {
        vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
        uv *= invAtan;
        uv += 0.5;
        return uv;
    }
    
    void main()
    {		
        vec2 uv = SampleSphericalMap(normalize(localPos)); // make sure to normalize localPos
        vec3 color = texture(equirectangularMap, uv).rgb;
        
        FragColor = vec4(color, 1.0);
    }
    """

    pbr_vertex_shader = """
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec2 aTexCoords;
    layout (location = 2) in vec3 aNormal;
    
    out vec2 TexCoords;
    out vec3 WorldPos;
    out vec3 Normal;
    
    uniform mat4 projection;
    uniform mat4 view;
    uniform mat4 model;
    
    void main()
    {
        TexCoords = aTexCoords;
        WorldPos = vec3(model * vec4(aPos, 1.0));
        Normal = mat3(model) * aNormal;   
    
        gl_Position =  projection * view * vec4(WorldPos, 1.0);
    }
    """

    pbr_fragment_shader = """
    #version 330 core
    out vec4 FragColor;
    in vec3 WorldPos;
    
    uniform samplerCube environmentMap;
    
    void main()
    {		
        vec3 envColor = texture(environmentMap, WorldPos).rgb;
        
        // HDR tonemap and gamma correct
        envColor = envColor / (envColor + vec3(1.0));
        envColor = pow(envColor, vec3(1.0/2.2)); 
        
        FragColor = vec4(envColor, 1.0);
    }
    """

    background_vertex_shader = """
    #version 330 core
    layout (location = 0) in vec3 aPos;
    
    uniform mat4 projection;
    uniform mat4 view;
    
    out vec3 WorldPos;
    
    void main()
    {
        WorldPos = aPos;
    
        mat4 rotView = mat4(mat3(view));
        vec4 clipPos = projection * rotView * vec4(WorldPos, 1.0);
    
        gl_Position = clipPos.xyww;
    }
    """

    background_fragment_shader = """
    #version 330 core
    out vec4 FragColor;
    in vec3 WorldPos;
    
    uniform samplerCube environmentMap;
    
    void main()
    {		
        vec3 envColor = texture(environmentMap, WorldPos).rgb;
        
        // HDR tonemap and gamma correct
        envColor = envColor / (envColor + vec3(1.0));
        envColor = pow(envColor, vec3(1.0/2.2)); 
        
        FragColor = vec4(envColor, 1.0);
    }
    """

    equirectangularToCubemapShader = Shader(cubemap_vertex_shader, cubemap_fragment_shader)
    pbrShader = Shader(pbr_vertex_shader, pbr_fragment_shader)
    backgroundShader = Shader(background_vertex_shader, background_fragment_shader)

    pbrShader.use()
    pbrShader.set_vec3("albedo", glm.vec3(0.5, 0.0, 0.0))
    pbrShader.set_float("ao", 1.0)

    backgroundShader.use()
    backgroundShader.set_int("environmentMap", 0)

    # lights
    # ------
    lightPositions = [glm.vec3(-10.0, 10.0, 10.0),
                      glm.vec3(10.0, 10.0, 10.0),
                      glm.vec3(-10.0, -10.0, 10.0),
                      glm.vec3(10.0, -10.0, 10.0)
                      ]

    lightColor = [glm.vec3(300.0, 300.0, 300.0),
                  glm.vec3(300.0, 300.0, 300.0),
                  glm.vec3(300.0, 300.0, 300.0),
                  glm.vec3(300.0, 300.0, 300.0)]

    nrRows = 7
    nrColumns = 7
    spacing = 2.5

    captureFBO = glGenFramebuffers(1)
    captureRBO = glGenRenderbuffers(1)

    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO)
    glBindRenderbuffer(GL_RENDERBUFFER, captureRBO)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 512, 512)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, captureRBO)

    # pbr: setup cubemap to render to and attach to framebuffer
    envCubemap = glGenTextures(1)
    glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap)
    for i in range(6):
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, 512, 512, 0, GL_RGB, GL_FLOAT, None)

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    # pbr: set up projection and view matrices for capturing data onto the 6 cubemap face directions
    captureProjection = glm.perspective(glm.radians(90.0), 1.0, 0.1, 10.0)
    captureViews = [glm.lookAt(glm.vec3(0.0, 0.0, 0.0), glm.vec3(1.0, 0.0, 0.0), glm.vec3(0.0, -1.0, 0.0)),
                    glm.lookAt(glm.vec3(0.0, 0.0, 0.0), glm.vec3(-1.0, 0.0, 0.0), glm.vec3(0.0, -1.0, 0.0)),
                    glm.lookAt(glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 1.0, 0.0), glm.vec3(0.0, 0.0, -1.0)),
                    glm.lookAt(glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, -1.0, 0.0), glm.vec3(0.0, 0.0, -1.0)),
                    glm.lookAt(glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 0.0, 1.0), glm.vec3(0.0, -1.0, 0.0)),
                    glm.lookAt(glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 0.0, -1.0), glm.vec3(0.0, -1.0, 0.0))]

    # pbr: convert HDR equirectangular environment map to cubemap equivalent
    equirectangularToCubemapShader.use()
    equirectangularToCubemapShader.set_int("equirectangularMap", 0)
    equirectangularToCubemapShader.set_mat4("projection", captureProjection)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, hdrTexture)

    glViewport(0, 0, screen_width, screen_height)
    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO)
    cubeVBO = []

    for i in range(6):
        equirectangularToCubemapShader.set_mat4("view", captureViews[i])
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, envCubemap, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        renderCube(cubeVBO)

    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    # initialize static shader uniforms before rendering
    projection = glm.perspective(glm.radians(45.0), screen_width / screen_height, 0.1, 100.0)
    pbrShader.use()
    pbrShader.set_mat4("projection", projection)
    backgroundShader.use()
    backgroundShader.set_mat4("projection", projection)

    # then before rendering, configure the viewport to the original framebuffer's screen dimensions
    scrWidth, scrHeight = glfw.get_framebuffer_size(window)
    glViewport(0, 0, scrWidth, scrHeight)
    skyboxVBO = []

    # render loop
    while not glfw.window_should_close(window):
        process_input(window)

        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # render scene, supplying the convoluted irradiance map to the final shader.
        pbrShader.use()
        view = camera.GetViewMatrix()
        pbrShader.set_mat4("view", view)
        pbrShader.set_vec3("camPos", camera.position)

        # for row in range(nrRows):
        #     pbrShader.set_float("metallic", row/nrRows)
        #     for col in range(nrColumns):
        #         pbrShader.set_float("roughness", glm.clamp(col/nrColumns, 0.05, 1.0))
        #
        #         model = glm.mat4()
        #         model = glm.translate(model, glm.vec3((col-nrColumns/2)*spacing,
        #                                               (row-nrRows*2)*spacing,
        #                                               -2.0))
        #         pbrShader.set_mat4("model", model)
        #         renderSphere()

        # render skybox(render as last to prevent overdraw)
        backgroundShader.use()
        backgroundShader.set_mat4("view", view)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap)
        renderCube(skyboxVBO)

        # glfw: swap buffers and poll IO events(keys pressed / released, mouse movedetc.)
        # -------------------------------------------------------------------------------
        glfw.swap_buffers(window)
        glfw.poll_events()




if __name__ == "__main__":
    main()