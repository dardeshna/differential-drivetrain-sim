import numpy as np
from scipy.integrate import solve_ivp

class Drivetrain():
    """Drivetrain model

    state: [x, y, theta, s_l, s_r, v_l, v_r]

    x = robot x position
    y = robot y position
    theta = robot heading
    s_l = left wheel distance
    s_r = right wheel distance
    v_l = left wheel speed
    v_r = right wheel speed

    inputs: [V_l, V_r]

    V_l = left voltage
    V_r = right voltage

    equations of motion:

    x_dot = (v_l + v_r) / 2 * cos(theta)
    y_dot = (v_l + v_r) / 2 * sin(theta)
    theta_dot = (v_r - v_l) / (2 * r_b)
    s_l_dot = v_l
    s_r_dot = v_r
    v_l_dot = C3*C1*v_l + C4*C1*v_r + C3*C2*V_l + C4*C2*V_r
    v_r_dot = C4*C1*v_l + C3*C1*v_r + C4*C2*V_l + C3*C2*V_r
    
    constants:

    C1 = -(G ** 2) * k_t / (1 / k_b * R * r ** 2)
    C2 = G * k_t / (R * r)
    C3 = 1 / m + r_b**2 / J
    C4 = 1 / m - r_b**2 / J

    R = motor resistance (multiple motors on same side combine in parallel)
    k_t = motor torque constant
    k_b = motor back-emf constant
    G = drivetrain gear ratio
    r = wheel radius
    r_b = base radius
    m = robot mass
    J = robot moment of inertia
    """

    def __init__(self, motor, m, r, r_b, J, G):
    
        C1 = -(G ** 2) * motor.k_t / (1 / motor.k_b * motor.R * r ** 2)
        C2 = G * motor.k_t / (motor.R * r)

        C3 = 1 / m + r_b**2 / J
        C4 = 1 / m - r_b**2 / J

        self.A = np.array([[C3 * C1, C4 * C1],
                    [C4 * C1, C3 * C1]])
        self.B = np.array([[C3 * C2, C4 * C2],
                    [C4 * C2, C3 * C2]])

        self.motor = motor
        self.m = m
        self.r = r
        self.r_b = r_b
        self.J = J
        self.G = G

    def f(self, x, u):

        x, y, theta, s_left, s_right, v_left, v_right = x

        v = (v_left + v_right) / 2
        a_left, a_right = self.A @ np.array([v_left, v_right]) + self.B @ u

        return np.array([
            v * np.cos(theta),
            v * np.sin(theta),
            (v_right - v_left) / (2.0 * self.r_b),
            v_left,
            v_right,
            a_left,
            a_right,
        ])

class DrivetrainSim():

    def __init__(self, drivetrain, x_0):

        self.drivetrain = drivetrain
        self.reset(x_0)
        
    def reset(self, x_0):

        self.xs = [x_0]
        self.us = [np.array([np.nan, np.nan])]
        self.ts = [0]

        self.i = 0

    def step(self, u, dt):

        self.us[self.i] = u

        # integrate equations of motion from t to t+dt with constant input u
        res = solve_ivp(lambda t, y: self.drivetrain.f(y, u), (self.ts[-1], self.ts[-1]+dt), self.xs[-1])

        self.xs.append(res['y'][:,-1])
        self.us.append(np.array([np.nan, np.nan]))
        self.ts.append(self.ts[-1]+dt)
        
        self.i += 1

    def solve(self, u, t):

        for i, dt in enumerate(np.diff(t)):
            self.step(u[i], dt)
