import glm
import numpy as np

class Camera:
    def __init__(self, position=glm.vec3(0.0, 0.0, 0.0), up=glm.vec3(0.0, 1.0, 0.0), yaw=-90.0, pitch=0.0,
                 front=glm.vec3(0.0, 0.0, -1.0)):
        self.position = position
        self.world_up = up
        self.yaw = yaw
        self.pitch = pitch
        self.front = front
        self.updateCameraVectors()

    def GetViewMatrix(self):
        return glm.lookAt(self.position, self.position + self.front, self.up)

    def updateCameraVectors(self):
        front_x = np.cos(glm.radians(self.yaw) * np.cos(glm.radians(self.pitch)))
        front_y = np.sin(glm.radians(self.pitch))
        front_z = np.sin(glm.radians(self.yaw) * np.cos(glm.radians(self.pitch)))
        self.front = glm.normalize(glm.vec3(front_x, front_y, front_z))

        self.right = glm.normalize(glm.cross(self.front, self.world_up))
        self.up = glm.normalize(glm.cross(self.right, self.front))
