# Henrique Weber, 2018
# adapted from https://learnopengl.com/Lighting/Colors
# This file is licensed under the MIT License.

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import glm


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

    vertices = [-0.5, -0.5, -0.5, 0.0, 0.0, -1.0,
                0.5, -0.5, -0.5, 0.0, 0.0, -1.0,
                0.5, 0.5, -0.5, 0.0, 0.0, -1.0,
                0.5, 0.5, -0.5, 0.0, 0.0, -1.0,
                -0.5, 0.5, -0.5, 0.0, 0.0, -1.0,
                -0.5, -0.5, -0.5, 0.0, 0.0, -1.0,

                -0.5, -0.5, 0.5, 0.0, 0.0, 1.0,
                0.5, -0.5, 0.5, 0.0, 0.0, 1.0,
                0.5, 0.5, 0.5, 0.0, 0.0, 1.0,
                0.5, 0.5, 0.5, 0.0, 0.0, 1.0,
                -0.5, 0.5, 0.5, 0.0, 0.0, 1.0,
                -0.5, -0.5, 0.5, 0.0, 0.0, 1.0,

                -0.5, 0.5, 0.5, -1.0, 0.0, 0.0,
                -0.5, 0.5, -0.5, -1.0, 0.0, 0.0,
                -0.5, -0.5, -0.5, -1.0, 0.0, 0.0,
                -0.5, -0.5, -0.5, -1.0, 0.0, 0.0,
                -0.5, -0.5, 0.5, -1.0, 0.0, 0.0,
                -0.5, 0.5, 0.5, -1.0, 0.0, 0.0,

                0.5, 0.5, 0.5, 1.0, 0.0, 0.0,
                0.5, 0.5, -0.5, 1.0, 0.0, 0.0,
                0.5, -0.5, -0.5, 1.0, 0.0, 0.0,
                0.5, -0.5, -0.5, 1.0, 0.0, 0.0,
                0.5, -0.5, 0.5, 1.0, 0.0, 0.0,
                0.5, 0.5, 0.5, 1.0, 0.0, 0.0,

                -0.5, -0.5, -0.5, 0.0, -1.0, 0.0,
                0.5, -0.5, -0.5, 0.0, -1.0, 0.0,
                0.5, -0.5, 0.5, 0.0, -1.0, 0.0,
                0.5, -0.5, 0.5, 0.0, -1.0, 0.0,
                -0.5, -0.5, 0.5, 0.0, -1.0, 0.0,
                -0.5, -0.5, -0.5, 0.0, -1.0, 0.0,

                -0.5, 0.5, -0.5, 0.0, 1.0, 0.0,
                0.5, 0.5, -0.5, 0.0, 1.0, 0.0,
                0.5, 0.5, 0.5, 0.0, 1.0, 0.0,
                0.5, 0.5, 0.5, 0.0, 1.0, 0.0,
                -0.5, 0.5, 0.5, 0.0, 1.0, 0.0,
                -0.5, 0.5, -0.5, 0.0, 1.0, 0.0
                ]
    vertices = np.array(vertices, dtype=np.float32)

    camera_target = glm.vec3(0.0, 0.0, 0.0)
    up = glm.vec3(0.0, 1.0, 0.0)
    camera_direction = glm.normalize(camera_pos - camera_target)
    camera_right = glm.normalize(glm.cross(up, camera_direction))
    camera_up = glm.cross(camera_direction, camera_right)

    projection = glm.perspective(glm.radians(45.0), screen_width / screen_height, 0.1, 100.0)

    light_vertex_shader = """
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;
    
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    
    out vec3 FragPos;  
    out vec3 Normal;
    
    void main()
    {
        gl_Position = projection * view * model * vec4(aPos, 1.0);
        FragPos = vec3(model * vec4(aPos, 1.0));
        Normal = aNormal;
    }
    """

    vertex_shader = """
    #version 330 core
    layout (location = 0) in vec3 aPos;
    
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    
    void main()
    {
        gl_Position = projection * view * model * vec4(aPos, 1.0);
    }
    """

    light_fragment_shader = """
    #version 330 core
    out vec4 FragColor;
      
    uniform vec3 objectColor;
    uniform vec3 lightColor;
    uniform vec3 lightPos;
    
    in vec3 Normal;
    in vec3 FragPos;    
    
    void main()
    {
        float ambientStrength = 0.1;
        vec3 ambient = ambientStrength * lightColor;
    
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(lightPos - FragPos);
        
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * lightColor;
        
        vec3 result = (ambient + diffuse) * objectColor;
        FragColor = vec4(result, 1.0);
    }
    """

    fragment_shader = """
    #version 330 core
    out vec4 FragColor;
    
    void main()
    {
        FragColor = vec4(1.0); // set all 4 vector values to 1.0
    }
    """

    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))

    light_shader = OpenGL.GL.shaders.compileProgram(
        OpenGL.GL.shaders.compileShader(light_vertex_shader, GL_VERTEX_SHADER),
        OpenGL.GL.shaders.compileShader(light_fragment_shader, GL_FRAGMENT_SHADER))

    # cube's VAO and VBO
    VBO = glGenBuffers(1)  # vertex buffer object, which stores vertices in the GPU's memory.
    cubeVAO = glGenVertexArrays(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # now, all calls will configure VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glBindVertexArray(cubeVAO)

    # position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * np.dtype(np.float32).itemsize, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    # color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * np.dtype(np.float32).itemsize,
                          ctypes.c_void_p(3 * np.dtype(np.float32).itemsize))
    glEnableVertexAttribArray(1)

    # light's VAO
    lightVAO = glGenVertexArrays(1)
    glBindVertexArray(lightVAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)

    # glBindBuffer(GL_ARRAY_BUFFER, 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * np.dtype(np.float32).itemsize, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    # glBindVertexArray(0)

    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)
    glEnable(GL_DEPTH_TEST)

    # render loop
    while not glfw.window_should_close(window):
        # per-frame time logic
        current_frame = glfw.get_time()
        delta_time = current_frame - last_frame
        last_frame = current_frame

        # input
        process_input(window)

        # render
        glClearColor(0.1, 0.1, 0.1, 1.0)  # state-setting function
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # state-using function

        glUseProgram(light_shader)
        objectColorLoc = glGetUniformLocation(light_shader, "objectColor")
        object_color = glm.vec3(1.0, 0.5, 0.31)
        glUniform3fv(objectColorLoc, 1, glm.value_ptr(object_color))

        lightLoc = glGetUniformLocation(light_shader, "lightColor")
        light_color = glm.vec3(1.0, 1.0, 1.0)
        glUniform3fv(lightLoc, 1, glm.value_ptr(light_color))
        lightPosLoc = glGetUniformLocation(light_shader, "lightPos")
        glUniform3fv(lightPosLoc, 1, glm.value_ptr(light_pos))

        # view/projection transformations
        projectionLoc = glGetUniformLocation(light_shader, "projection")
        glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm.value_ptr(projection))
        view = glm.lookAt(camera_pos, camera_pos + camera_front, camera_up)
        viewLoc = glGetUniformLocation(light_shader, "view")
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm.value_ptr(view))

        # world transformations
        model = glm.mat4(1.0)
        modelLoc = glGetUniformLocation(light_shader, "model")
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm.value_ptr(model))

        # render the cube
        glBindVertexArray(cubeVAO)
        glDrawArrays(GL_TRIANGLES, 0, 36)

        # also draw the lamp object
        glUseProgram(shader)
        projectionLoc = glGetUniformLocation(shader, "projection")
        glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm.value_ptr(projection))
        viewLoc = glGetUniformLocation(shader, "view")
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm.value_ptr(view))

        model = glm.mat4(1.0)
        model = glm.translate(model, light_pos)
        model = glm.scale(model, glm.vec3(0.2))
        modelLoc = glGetUniformLocation(shader, "model")
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm.value_ptr(model))
        glBindVertexArray(lightVAO)
        glDrawArrays(GL_TRIANGLES, 0, 36)

        # check and call events and swap the buffers
        glfw.swap_buffers(window)
        glfw.poll_events()
    glfw.terminate()


if __name__ == "__main__":
    main()
