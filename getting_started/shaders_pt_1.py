# Henrique Weber, 2018
# adapted from https://learnopengl.com/Getting-started/Hello-Triangle
# This file is licensed under the MIT License.

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np


def framebuffer_size_callback(window, width, height):
    glViewport(0, 0, width, height)


def process_input(window):
    if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
        glfw.set_window_should_close(window, True)


def main():
    # initialize glfw
    if not glfw.init():
        return

    window = glfw.create_window(800, 600, "LearnOpenGL", None, None)

    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    vertices = [-0.5, -0.5, 0.0,
                0.5, -0.5, 0.0,
                0.0, 0.5, 0.0]
    vertices = np.array(vertices, dtype=np.float32)

    vertex_shader = """
    #version 330 core
    layout (location = 0) in vec3 aPos; // the position variable has attribute position 0
      
    out vec4 vertexColor; // specify a color output to the fragment shader
    
    void main()
    {
        gl_Position = vec4(aPos, 1.0); // see how we directly give a vec3 to vec4's constructor
        vertexColor = vec4(0.5, 0.0, 0.0, 1.0); // set the output variable to a dark-red color
    }
    """

    fragment_shader = """
    #version 330 core
    out vec4 FragColor;
      
    in vec4 vertexColor; // the input variable from the vertex shader (same name and same type)  
    
    void main()
    {
        FragColor = vertexColor;
    }
    """

    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))

    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)  # vertex buffer object, which stores vertices in the GPU's memory.
    glBindVertexArray(VAO)

    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # now, all calls will configure VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)  # copy user-defined data into VBO

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * np.dtype(np.float32).itemsize, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glBindBuffer(GL_ARRAY_BUFFER, 0)

    glBindVertexArray(0)

    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)

    # render loop
    while not glfw.window_should_close(window):
        # input
        process_input(window)

        # rendering commands here
        glClearColor(0.2, 0.3, 0.3, 1.0)  # state-setting function
        glClear(GL_COLOR_BUFFER_BIT)  # state-using function

        # render triangle
        glUseProgram(shader)
        glBindVertexArray(VAO)
        glDrawArrays(GL_TRIANGLES, 0, 3)

        # check and call events and swap the buffers
        glfw.swap_buffers(window)
        glfw.poll_events()
    glfw.terminate()


if __name__ == "__main__":
    main()
