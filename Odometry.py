import numpy as np

class Odometry():

    def __init__(self, x_0):

        self.x, self.y, self.theta, self.s_left, self.s_right = x_0

    def update(self, s_left, s_right, theta):

        delta_s_left = s_left - self.s_left
        delta_s_right = s_right - self.s_right

        dx = (delta_s_left + delta_s_right) / 2
        dy = 0
        dtheta = (theta-self.theta + np.pi) % (2 * np.pi) - np.pi

        if abs(dtheta) < 1e-9:
            s = 1.0 - 1.0 / 6.0 * dtheta * dtheta
            c = 0.5 * dtheta
        else:
            s = np.sin(dtheta) / dtheta
            c = (1 - np.cos(dtheta)) / dtheta
        
        dx, dy = dx * s - dy * c, dx * c + dy * s

        self.x += dx * np.cos(self.theta) - dy * np.sin(self.theta)
        self.y += dx * np.sin(self.theta) + dy * np.cos(self.theta)
        self.theta = theta

        self.s_left = s_left
        self.s_right = s_right

    def get(self):

        return np.array([self.x, self.y, self.theta])