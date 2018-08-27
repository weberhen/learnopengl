# Henrique Weber, 2018
# adapted from https://learnopengl.com/Getting-started/Hello-Window
# This file is licensed under the MIT License.

import glfw
from OpenGL.GL import *


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

    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)

    # render loop
    while not glfw.window_should_close(window):
        # input
        process_input(window)

        # rendering commands here
        glClearColor(0.2, 0.3, 0.3, 1.0)  # state-setting function
        glClear(GL_COLOR_BUFFER_BIT)  # state-using function

        # check and call events and swap the buffers
        glfw.swap_buffers(window)
        glfw.poll_events()
    glfw.terminate()


if __name__ == "__main__":
    main()
