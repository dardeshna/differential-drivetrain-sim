class Motor():
    """Motor model

    V = T/k_t * R + k_b*w

    V = input voltage
    T = output torque
    w = motor speed

    R = motor resistance
    k_t = torque constant
    k_b = back-emf constant

    R = V_nominal / I_stall
    k_b = (V_nominal - R * I_free) / w_free
    k_t = T_stall / I_stall
    """

    def __init__(self, V_n, w_f, T_s, I_s, I_f, n_motors):

        self.V_n = V_n
        self.w_f = w_f
        self.T_s = T_s * n_motors
        self.I_s = I_s * n_motors
        self.I_f = I_f * n_motors

        self.R = self.V_n / self.I_s
        self.k_b = (self.V_n - self.R * self.I_f) / self.w_f
        self.k_t = self.T_s / self.I_s