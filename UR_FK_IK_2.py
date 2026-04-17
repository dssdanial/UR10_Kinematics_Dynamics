#Developing a robot manipulation using Forward and Inverse Kinematics from scratch. #@DssDanial 

import roboticstoolbox as rtb
import numpy as np
import spatialmath as sm
import swift
import time
import spatialgeometry as sg
import matplotlib.pyplot as plt

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
    theta=float(theta)
    d=float(d)
    a=float(a)
    alpha=float(alpha)
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0.0,              np.sin(alpha),                np.cos(alpha),               d],
        [0.0,              0.0,                            0.0,                           1.0]
    ])

def forward_kinematics(q, d, a, alpha):
    n_joint = len(q)
    T = np.zeros((n_joint, 4, 4))  # Store all transforms
    T_total = np.eye(4)



    #dh_base_offset = np.array([0, 0, 0.1273]).T  # UR10's d1
    

    for i in range(n_joint):
        T_i = dh_matrix(q[i], d[i], a[i], alpha[i])
        #T_i[2,3]=T_i[2,3]-0.127   # Apply base offset correction

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

   
    J = np.zeros((6, len(q)))
    for i in range(len(q)):
        J[0:3, i] = np.cross(z[i], o[-1] - o[i])    # linear velocity
        J[3:6, i] = z[i]                            # angular velocity

    return J



def skew_to_vec(skew_mat):
    return np.array([
        skew_mat[2,1],
        skew_mat[0,2],
        skew_mat[1,0]
    ])

# -------------------------------------------
# Inverse Kinematics (Position only)
# -------------------------------------------
def inverse_kinematics_numerical(q_init, T_goal, method):
    max_iter=20000
    alpha_step=0.55
    lam=0.05
    threshold=1e-3

    q = q_init.copy()
    errors, dq_norms, sigmas = [], [], []

    # Handle SE3 input or plain position vector
    if isinstance(T_goal, sm.SE3):
        T_goal = np.array(T_goal.t).flatten()
    else:
        T_goal = np.array(T_goal).flatten()


    
    for _ in range(max_iter):
        T_all = forward_kinematics(q, robot.d, robot.a, robot.alpha)
        pos_current = sm.SE3(T_all[-1])

        T_goal = sm.SE3(T_goal)
        T_err = pos_current.inv() * T_goal

        pos_err = T_err.t  # shape (3,)


        so3mat = sm.SO3(T_err.R).log()
        rot_err = skew_to_vec(so3mat)




        # so3mat = sm.SO3(T_err.R).log()  # 3x3 skew matrix
        # rot_err = sm.so3.mat2vec(so3mat)  # convert to 3-vector
        rot_err = np.array(rot_err).ravel()

        error_6d = np.hstack((pos_err, rot_err))  # shape (6,)


        J = Jacobian(q)  # full 6xN Jacobian expected here
        print("error_6d shape:", error_6d.shape)
        print("error_6d:", error_6d)
        
        
        # Choose solver
        if method == "pinv":
            dq = alpha_step * np.linalg.pinv(J) @ error_6d
        elif method == "dls":
            J_damped = J.T @ np.linalg.inv(J @ J.T + (lam**2) * np.eye(6))
            dq = alpha_step * J_damped @ error_6d
        
        # SVD for singularity measure
        U, S, Vt = np.linalg.svd(J)
        sigmas.append(np.min(S))
        dq_norms.append(np.linalg.norm(dq))
        q += dq

        if np.linalg.norm(error_6d) < threshold:
            print(f"Converged in {max_iter} iterations.")
            print("\nerrors: \n", error_6d)
            break
    
    return q, errors, dq_norms, sigmas



# -------------------------------------------
# Define Goal Pose and Trajectory
# -------------------------------------------
T_goal = sm.SE3(0.5, 0.2, 0.9) * sm.SE3.RPY([0.0, 0.0, 0])


theta_sol, errors_pinv, dq_pinv, sigma_pinv  = inverse_kinematics_numerical(q0,T_goal, "pinv")
trajectory = rtb.jtraj(q0, theta_sol, 50)

# Run both methods
q, errors_dls, dq_dls, sigma_dls = inverse_kinematics_numerical(q0, T_goal, "dls")

# -------------------------------------------
# Simulation with Swift
# -------------------------------------------
target = sg.Sphere(0.03, color=[0.1, 0.4, 0.2, 0.8], pose=T_goal)
target_frame = sg.Axes(0.2, pose=T_goal)

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

# Plot
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.plot(errors_pinv, label="Pinv")
plt.plot(errors_dls, label="DLS")
plt.xlabel("Iteration")
plt.ylabel("Error norm")
plt.legend()
plt.title("Task-space error")

plt.subplot(1,3,2)
plt.plot(dq_pinv, label="Pinv")
plt.plot(dq_dls, label="DLS")
plt.xlabel("Iteration")
plt.ylabel("||Δq||")
plt.legend()
plt.title("Joint motion per step")

plt.subplot(1,3,3)
plt.plot(sigma_pinv, label="Pinv")
plt.plot(sigma_dls, label="DLS")
plt.xlabel("Iteration")
plt.ylabel("Min singular value")
plt.legend()
plt.title("Closeness to singularity")

plt.tight_layout()
plt.show()
