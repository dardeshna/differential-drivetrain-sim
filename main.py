import copy

import numpy as np
from matplotlib import pyplot as plt

from Drivetrain import Drivetrain, DrivetrainSim
from Motor import Motor
from Odometry import Odometry
from TrajectoryGenerator import compute_smoothed_traj, modify_traj_with_limits
from TalonSRX import TalonSRX, TalonSRXControlMode
from Ramsete import Ramsete

SIM_DT = 0.001
CTRL_DT = 0.02
EPS = 1e-9
T_EXTRA = 1 # time to simulate past end of trajectory

# motor and drivetrain constants

V_nominal = 12 # V
w_free =  5676 * (2 * np.pi / 60) # RPM -> rad/s
T_stall = 2.6 # Nm
I_stall = 105 # A
I_free = 1.8 # A
n_motors_per_side = 2

m_robot = 45 # kg
J_robot = 3.3 # kg*m^2
r_wheel = 0.051 # m
r_base = 0.56 # m
G_ratio = 6

neo = Motor(V_nominal, w_free, T_stall, I_stall, I_free, n_motors_per_side)
drivetrain = Drivetrain(neo, m_robot, r_wheel, r_base, J_robot, G_ratio)

# xy waypoints for straight path
# path = np.array([
#     [0, 0],
#     [5, 0],
#     [10, 0],
#     [15, 0],
# ])

# xy waypoints for curved path
path = np.array([
    [-10, 0],
    [0, 0],
    [3, 0],
    [5, 2],
    [7, -3],
])

# generate a basic spline trajectory for demonstration

v_nominal = 3 # m/s
alpha = 0.5 # smoothing parameter

traj, traj_ts, = compute_smoothed_traj(path, v_nominal, alpha, CTRL_DT)
traj_ts, v_goal, w_goal, traj = modify_traj_with_limits(traj, traj_ts, 3, np.inf, CTRL_DT)

x_0 = traj[0, 0:3] # initial state

# construct drivetrain simulation, odometry and controllers

sim = DrivetrainSim(drivetrain, np.append(x_0, np.zeros(4)))
odom = Odometry(np.append(x_0, np.zeros(2)))

talon_base = TalonSRX()
ticks_per_meter, ticks_per_100ms_per_meter_per_second = talon_base.get_conversions(r_wheel)

talon_base.P = 0.015
talon_base.F = 1023 / (w_free / G_ratio * r_wheel * ticks_per_100ms_per_meter_per_second)

talon_l = talon_base
talon_r = copy.deepcopy(talon_base)

ramsete = Ramsete(b=2.0, zeta=0.8)

# set up timestamps and counters for simulation

sim_ts = np.arange(int((traj_ts[-1]+T_EXTRA)/SIM_DT)) * SIM_DT
ctrl_ts = np.arange(int((traj_ts[-1]+T_EXTRA)/CTRL_DT)) * CTRL_DT

update_ctrl = np.zeros_like(sim_ts, dtype=bool)
update_ctrl[np.searchsorted(sim_ts, ctrl_ts)] = True

j = 0

# run simulation

for i, t in enumerate(sim_ts):

    if update_ctrl[i]:

        odom.update(sim.xs[-1][3], sim.xs[-1][4], sim.xs[-1][2])

        if j < len(traj_ts):
            v_adj, w_adj = ramsete.calculate(traj[j, 0:3], v_goal[j], w_goal[j], odom.get())
        else:
            v_adj, w_adj = ramsete.calculate(traj[-1, 0:3], 0, 0, odom.get())
        
        # v_adj, w_adj = v_goal[j], w_goal[j]

        v_l = v_adj - w_adj * r_base
        v_r = v_adj + w_adj * r_base

        talon_l.set(TalonSRXControlMode.Velocity, v_l * ticks_per_100ms_per_meter_per_second)
        talon_r.set(TalonSRXControlMode.Velocity, v_r * ticks_per_100ms_per_meter_per_second)

        j += 1

    talon_l.update()
    talon_r.update()

    V_l = talon_l.getMotorOutputVoltage()
    V_r = talon_r.getMotorOutputVoltage()

    sim.step((V_l, V_r), SIM_DT)

    talon_l.pushReading(sim.xs[-1][3] * ticks_per_meter)
    talon_r.pushReading(sim.xs[-1][4] * ticks_per_meter)

xs = np.array(sim.xs)
us = np.array(sim.us)

# plot robot pose

plt.figure()
plt.plot(traj[:, 0], traj[:, 1], label='trajectory')
plt.plot(xs[:, 0], xs[:, 1], label='robot')
plt.title('robot pose')
plt.legend()

# plot wheel speeds

v_l_goal = v_goal - w_goal * r_base
v_r_goal = v_goal + w_goal * r_base

plt.figure()
plt.plot(traj_ts, v_l_goal, label='left goal')
plt.plot(traj_ts, v_r_goal, label='right goal')

plt.plot(sim.ts, xs[:,5], label='left actual')
plt.plot(sim.ts, xs[:,6], label='right actual')
plt.title('wheel speeds')
plt.legend()

# plot drive voltages

plt.figure()
plt.plot(sim.ts, us[:, 0], label='left voltage')
plt.plot(sim.ts, us[:, 1], label='right voltage')
plt.title('control inputs')
plt.legend()

plt.show()