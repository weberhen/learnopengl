import glm
from OpenGL.GL import *


class Shader:
    def __init__(self, vertex_shader_data, fragment_shader_data):
        # vertex shader
        vertex = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex, vertex_shader_data)

        glCompileShader(vertex)
        if glGetShaderiv(vertex, GL_COMPILE_STATUS) != GL_TRUE:
            raise RuntimeError(glGetShaderInfoLog(vertex))

        # fragment shader
        fragment = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment, fragment_shader_data)
        glCompileShader(fragment)
        if glGetShaderiv(fragment, GL_COMPILE_STATUS) != GL_TRUE:
            raise RuntimeError(glGetShaderInfoLog(fragment))

        #shader program
        self.ID = glCreateProgram()
        glAttachShader(self.ID, vertex)
        glAttachShader(self.ID, fragment)
        glLinkProgram(self.ID)
        if glGetProgramiv(self.ID, GL_LINK_STATUS) != GL_TRUE:
            raise RuntimeError(glGetProgramInfoLog(self.ID))
        # delete the shaders as they're linked into our program now and no longer necessery
        glDeleteShader(vertex)
        glDeleteShader(fragment)

    def use(self):
        glUseProgram(self.ID)

    def set_bool(self, name, value):
        glUniform1i(glGetUniformLocation(self.ID, name), value)

    def set_int(self, name, value):
        glUniform1i(glGetUniformLocation(self.ID, name), value)

    def set_float(self, name, value):
        glUniform1f(glGetUniformLocation(self.ID, name), value)

    def set_vec2(self, name, value):
        glUniform2fv(glGetUniformLocation(self.ID, name), value)

    def set_vec3(self, name, value):
        glUniform3fv(glGetUniformLocation(self.ID, name), 1, glm.value_ptr(value))

    def set_vec4(self, name, value):
        glUniform4fv(glGetUniformLocation(self.ID, name), value)

    def set_mat4(self, name, mat):
        glUniformMatrix4fv(glGetUniformLocation(self.ID, name), 1, GL_FALSE, glm.value_ptr(mat))


    @staticmethod
    def check_compile_errors(shader, type):
        if type == 'PROGRAM':
            success = glGetShaderiv(shader, GL_COMPILE_STATUS)
            if not success:
                infolog = glGetShaderInfoLog(shader, 1024, None)
                print('ERROR::SHADER_COMPILATION_ERROR of type: {} \n {}'.format(type, infolog))
        else:
            success = glGetProgramiv(shader, GL_LINK_STATUS)
            if not success:
                infolog = glGetProgramInfoLog(shader, 1024, None)
                print('ERROR::PROGRAM_LINKING_ERROR of type: {} \n {}'.format(type, infolog))

