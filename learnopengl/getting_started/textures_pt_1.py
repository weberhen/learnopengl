# Henrique Weber, 2018
# adapted from https://learnopengl.com/Getting-started/Textures
# This file is licensed under the MIT License.

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
from PIL.Image import open


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

    vertices = [0.5, 0.5, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                0.5, -0.5, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                -0.5, -0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                -0.5, 0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    vertices = np.array(vertices, dtype=np.float32)

    indices = [0, 1, 3,
               1, 2, 3]
    indices = np.array(indices, dtype=np.uint32)

    im = open('../textures/container.jpg')
    width, height, image = im.size[0], im.size[1], im.tobytes("raw", "RGB", 0, -1)
    im.close()
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image)
    glGenerateMipmap(GL_TEXTURE_2D)

    vertex_shader = """
    #version 330 core
    layout (location = 0) in vec3 aPos;   // the position variable has attribute position 0
    layout (location = 1) in vec3 aColor; // the color variable has attribute position 1
    layout (location = 2) in vec2 aTexCoord;
      
    out vec3 ourColor; // output a color to the fragment shader
    out vec2 TexCoord;
    
    void main()
    {
        gl_Position = vec4(aPos, 1.0);
        ourColor = aColor; // set ourColor to the input color we got from the vertex data
        TexCoord = aTexCoord;
    }  
    """

    fragment_shader = """
    #version 330 core
    out vec4 FragColor;  
    in vec3 ourColor;
    in vec2 TexCoord;
    
    uniform sampler2D ourTexture;
      
    void main()
    {
        // FragColor = texture(ourTexture, TexCoord);
        FragColor = texture(ourTexture, TexCoord) * vec4(ourColor, 1.0);
    }
    """

    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))

    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)  # vertex buffer object, which stores vertices in the GPU's memory.
    EBO = glGenBuffers(1)  # Element Buffer Object
    glBindVertexArray(VAO)

    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # now, all calls will configure VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)  # copy user-defined data into VBO

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    # position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * np.dtype(np.float32).itemsize,
                          ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    # color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * np.dtype(np.float32).itemsize,
                          ctypes.c_void_p(3 * np.dtype(np.float32).itemsize))
    glEnableVertexAttribArray(1)

    # texture attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * np.dtype(np.float32).itemsize,
                          ctypes.c_void_p(6 * np.dtype(np.float32).itemsize))
    glEnableVertexAttribArray(2)

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
        timeValue = glfw.get_time()
        greenValue = (np.sin(timeValue) / 2.0) + 0.5
        vertexColorLocation = glGetUniformLocation(shader, "ourColor")
        glUseProgram(shader)
        glUniform4f(vertexColorLocation, 0.0, greenValue, 0.0, 1.0)
        glBindTexture(GL_TEXTURE_2D, texture)
        glBindVertexArray(VAO)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

        # check and call events and swap the buffers
        glfw.swap_buffers(window)
        glfw.poll_events()
    glfw.terminate()


if __name__ == "__main__":
    main()
