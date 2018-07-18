# Henrique Weber, 2018
# adapted from https://learnopengl.com/Getting-started/Camera
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


def process_input(window):
    if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
        glfw.set_window_should_close(window, True)

    global camera_pos, camera_front, camera_up
    camera_speed = 0.05
    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
        camera_pos += camera_speed * camera_front
    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
        camera_pos -= camera_speed * camera_front
    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
        camera_pos -= glm.normalize(glm.cross(camera_front, camera_up)) * camera_speed
    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
        camera_pos += glm.normalize(glm.cross(camera_front, camera_up)) * camera_speed


def main():
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

    vertices = [-0.5, -0.5, -0.5, 0.0, 0.0, 1.0,
                0.5, -0.5, -0.5, 1.0, 0.0, 1.0,
                0.5, 0.5, -0.5, 1.0, 1.0, 1.0,
                0.5, 0.5, -0.5, 1.0, 1.0, 1.0,
                -0.5, 0.5, -0.5, 0.0, 1.0, 1.0,
                -0.5, -0.5, -0.5, 0.0, 0.0, 1.0,

                -0.5, -0.5, 0.5, 0.0, 0.0, 1.0,
                0.5, -0.5, 0.5, 1.0, 0.0, 1.0,
                0.5, 0.5, 0.5, 1.0, 1.0, 1.0,
                0.5, 0.5, 0.5, 1.0, 1.0, 1.0,
                -0.5, 0.5, 0.5, 0.0, 1.0, 1.0,
                -0.5, -0.5, 0.5, 0.0, 0.0, 1.0,

                -0.5, 0.5, 0.5, 1.0, 0.0, 1.0,
                -0.5, 0.5, -0.5, 1.0, 1.0, 1.0,
                -0.5, -0.5, -0.5, 0.0, 1.0, 1.0,
                -0.5, -0.5, -0.5, 0.0, 1.0, 1.0,
                -0.5, -0.5, 0.5, 0.0, 0.0, 1.0,
                -0.5, 0.5, 0.5, 1.0, 0.0, 1.0,

                0.5, 0.5, 0.5, 1.0, 0.0, 1.0,
                0.5, 0.5, -0.5, 1.0, 1.0, 1.0,
                0.5, -0.5, -0.5, 0.0, 1.0, 1.0,
                0.5, -0.5, -0.5, 0.0, 1.0, 1.0,
                0.5, -0.5, 0.5, 0.0, 0.0, 1.0,
                0.5, 0.5, 0.5, 1.0, 0.0, 1.0,

                -0.5, -0.5, -0.5, 0.0, 1.0, 1.0,
                0.5, -0.5, -0.5, 1.0, 1.0, 1.0,
                0.5, -0.5, 0.5, 1.0, 0.0, 1.0,
                0.5, -0.5, 0.5, 1.0, 0.0, 1.0,
                -0.5, -0.5, 0.5, 0.0, 0.0, 1.0,
                -0.5, -0.5, -0.5, 0.0, 1.0, 1.0,

                -0.5, 0.5, -0.5, 0.0, 1.0, 1.0,
                0.5, 0.5, -0.5, 1.0, 1.0, 1.0,
                0.5, 0.5, 0.5, 1.0, 0.0, 1.0,
                0.5, 0.5, 0.5, 1.0, 0.0, 1.0,
                -0.5, 0.5, 0.5, 0.0, 0.0, 1.0,
                -0.5, 0.5, -0.5, 0.0, 1.0, 1.0
                ]
    vertices = np.array(vertices, dtype=np.float32)

    camera_target = glm.vec3(0.0, 0.0, 0.0)
    up = glm.vec3(0.0, 1.0, 0.0)
    camera_direction = glm.normalize(camera_pos - camera_target)
    camera_right = glm.normalize(glm.cross(up, camera_direction))
    camera_up = glm.cross(camera_direction, camera_right)

    cube_positions = [
        glm.vec3(0.0, 0.0, 0.0),
        glm.vec3(2.0, 5.0, -15.0),
        glm.vec3(-1.5, -2.2, -2.5),
        glm.vec3(-3.8, -2.0, -12.3),
        glm.vec3(2.4, -0.4, -3.5),
        glm.vec3(-1.7, 3.0, -7.5),
        glm.vec3(1.3, -2.0, -2.5),
        glm.vec3(1.5, 2.0, -2.5),
        glm.vec3(1.5, 0.2, -1.5),
        glm.vec3(-1.3, 1.0, -1.5)
    ]

    # view = glm.mat4(1.0)
    # view = glm.translate(view, glm.vec3(0.0, 0.0, -3.0))


    projection = glm.perspective(glm.radians(45.0), screen_width / screen_height, 0.1, 100.0)

    vertex_shader = """
    #version 330 core
    layout (location = 0) in vec3 aPos;   // the position variable has attribute position 0
    layout (location = 1) in vec3 aColor; // the color variable has attribute position 1
      
    out vec3 ourColor; // output a color to the fragment shader
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    
    void main()
    {
        gl_Position = projection * view * model * vec4(aPos, 1.0);
        ourColor = aColor; // set ourColor to the input color we got from the vertex data
    }  
    """

    fragment_shader = """
    #version 330 core
    out vec4 FragColor;  
    in vec3 ourColor;
      
    void main()
    {
        FragColor = vec4(ourColor, 1.0);
    }
    """

    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))

    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)  # vertex buffer object, which stores vertices in the GPU's memory.
    glBindVertexArray(VAO)

    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # now, all calls will configure VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)  # copy user-defined data into VBO

    # position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * np.dtype(np.float32).itemsize, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    # color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * np.dtype(np.float32).itemsize,
                          ctypes.c_void_p(3 * np.dtype(np.float32).itemsize))
    glEnableVertexAttribArray(1)

    glBindBuffer(GL_ARRAY_BUFFER, 0)

    glBindVertexArray(0)

    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)
    glEnable(GL_DEPTH_TEST)

    # render triangle
    glUseProgram(shader)

    projectionLoc = glGetUniformLocation(shader, "projection")
    glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm.value_ptr(projection))

    # render loop
    while not glfw.window_should_close(window):
        # input
        process_input(window)

        # rendering commands here

        glClearColor(0.2, 0.3, 0.3, 1.0)  # state-setting function
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # state-using function

        view = glm.lookAt(camera_pos, camera_pos + camera_front, camera_up)
        viewLoc = glGetUniformLocation(shader, "view")
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm.value_ptr(view))

        for i in range(10):
            glDrawArrays(GL_TRIANGLES, 0, 36)

            model = glm.mat4(1.0)
            model = glm.translate(model, cube_positions[i])
            angle = 20.0 * i
            model = glm.rotate(model, glm.radians(angle), glm.vec3(1.0, 0.3, 0.5))
            modelLoc = glGetUniformLocation(shader, "model")
            glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm.value_ptr(model))

        glBindVertexArray(VAO)

        # check and call events and swap the buffers
        glfw.swap_buffers(window)
        glfw.poll_events()
    glfw.terminate()


if __name__ == "__main__":
    main()
