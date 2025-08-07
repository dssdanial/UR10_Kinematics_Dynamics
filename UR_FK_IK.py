#Developing a robot manipulation using Forward and Inverse Kinematics from scratch. #@DssDanial 

import roboticstoolbox as rtb
import numpy as np
import spatialmath as sm
import swift
import time
import spatialgeometry as sg

# Load DH-based UR10 model
robot = rtb.models.DH.UR10()
print(robot)

# Load full URDF-based UR10 model for simulation
ur10 = rtb.models.UR10()
ur10.q = [0, -np.pi / 2, 0, 0, 0, 0]
q0 = ur10.q.copy()

# -------------------------------------------
# Forward Kinematics using DH parameters
# -------------------------------------------
def dh_matrix(theta, d, a, alpha):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])

def forward_kinematics(q, d, a, alpha):
    n_joint = len(q)
    T = np.zeros((n_joint, 4, 4))  # Store all transforms
    T_total = np.eye(4)

    for i in range(n_joint):
        T_i = dh_matrix(q[i], d[i], a[i], alpha[i])
        T_total = T_total @ T_i
        T[i] = T_total

    return T

# -------------------------------------------
# Jacobian Computation (in base frame)
# -------------------------------------------
def Jacobian(q):
    T = np.eye(4)
    z = [np.array([0, 0, 1])]
    o = [np.array([0, 0, 0])]

    TF_lists=forward_kinematics(q, robot.d, robot.a, robot.alpha)

    for i in range(len(q)):
        T = TF_lists[i]
        z.append(T[0:3, 2])
        o.append(T[0:3, 3])

    # Apply base offset correction
    dh_base_offset = np.array([0, 0, 0.1273])  # UR10's d1
    o = [oi - dh_base_offset for oi in o]

    J = np.zeros((6, len(q)))
    for i in range(len(q)):
        J[0:3, i] = np.cross(z[i], o[-1] - o[i])    # linear velocity
        J[3:6, i] = z[i]                            # angular velocity

    return J




# -------------------------------------------
# Inverse Kinematics (Position only)
# -------------------------------------------
def inverse_kinematics_numerical(q_desired, q_init):
    max_iter=100
    tol=1e-3
    alpha_step=0.5
    
    if hasattr(q_desired, 't'):
        q_desired = np.array(q_desired.t) #This is because the structure of T_goal has the attribute .t

    q = q_init.copy()

    for i in range(max_iter):
        T_all = forward_kinematics(q, robot.d, robot.a, robot.alpha)
        q_current = T_all[-1][:3, 3]
        error = q_desired - q_current

        if np.linalg.norm(error) < tol:
            print(f"Converged in {i} iterations.")
            break

        J = Jacobian(q)
        J_pos = J[0:3, :]
        dq = alpha_step * np.linalg.pinv(J_pos) @ error
        q += dq

    return q




def compare_jacobians(q):
    # Your computed Jacobian
    J_custom = Jacobian(q)
    
    # Library Jacobian (base frame)
    J_library = robot.jacob0(q)

    # Print both
    print("Custom Jacobian:\n", np.round(J_custom, 4))
    print("\nLibrary Jacobian (jacob0):\n", np.round(J_library, 4))

    # Compare the difference
    diff = J_custom - J_library
    print("\nDifference (Custom - Library):\n", np.round(diff, 4))
    print("\nMax absolute error:", np.max(np.abs(diff)))



# -------------------------------------------
# Define Goal Pose and Trajectory
# -------------------------------------------
T_goal = sm.SE3(0.5, 0.2, 0.9) * sm.SE3.RPY([np.pi/2, np.pi, 0])
theta_sol = inverse_kinematics_numerical(T_goal, q0)
trajectory = rtb.jtraj(q0, theta_sol, 50)



# -------------------------------------------
# Simulation with Swift
# -------------------------------------------
target = sg.Sphere(0.03, color=[0.1, 0.4, 0.2, 0.8], base=T_goal)
target_frame = sg.Axes(0.2, base=T_goal)

sim = swift.Swift()
sim.launch(realtime=True)
sim.add(ur10)
sim.add(target)
sim.add(target_frame)
sim.set_camera_pose([2.4, 2.0, 0.7], [0.0, 0.0, 1.0])

for q in trajectory.q:
    ur10.q = q
    sim.step()
    time.sleep(0.05)
