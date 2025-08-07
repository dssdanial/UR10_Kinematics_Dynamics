# 📄 Robot Kinematics – DH Parameters, Forward & Inverse Kinematics

## 📌 Overview

This project implements and demonstrates **robot kinematics** for a 6-DOF manipulator (e.g., UR10) using:
- Denavit–Hartenberg (DH) parameters
- Forward Kinematics (FK)
- Jacobian-based Inverse Kinematics (IK)

The code builds all components **from scratch**, without relying on prebuilt kinematic libraries, and culminates in solving for the joint angles that reach a desired end-effector pose.

---

## 📐 1. DH Parameters

The Denavit–Hartenberg convention describes a robot's kinematic chain via 4 parameters per joint:

- `θᵢ`: joint angle (variable for revolute joints)
- `dᵢ`: offset along previous z-axis
- `aᵢ`: length of the common normal (x-axis distance)
- `αᵢ`: twist angle between z-axes

Each link's transformation matrix is:

```
T_i =
[[cosθᵢ, -sinθᵢcosαᵢ,  sinθᵢsinαᵢ, aᵢcosθᵢ],
 [sinθᵢ,  cosθᵢcosαᵢ, -cosθᵢsinαᵢ, aᵢsinθᵢ],
 [0,       sinαᵢ,        cosαᵢ,       dᵢ],
 [0,         0,             0,         1]]
```

---

## 🤖 2. Forward Kinematics (FK)

FK computes the transformation from the base frame to the end-effector:

```python
def forward_kinematics(q, d, a, alpha):
    T = np.eye(4)
    transforms = []
    for i in range(len(q)):
        A = dh_matrix(q[i], d[i], a[i], alpha[i])
        T = T @ A
        transforms.append(T)
    return transforms  # T_0_1, ..., T_0_n
```

Where `dh_matrix(...)` computes the individual transformation using the matrix above.

---

## 📉 3. Jacobian

The **geometric Jacobian** relates joint velocities to end-effector twist:

```
ẋ = J(q) · q̇
```

For revolute joints:

```
J_vᵢ = zᵢ × (oₙ - oᵢ)
J_ωᵢ = zᵢ
```

Where:
- `zᵢ`: rotation axis of joint i (in base frame)
- `oᵢ`: position of joint i
- `oₙ`: position of end-effector

---

## 🔁 4. Inverse Kinematics (IK)

### ✅ Position-Only IK (Numerical)

This simplified solver ignores orientation and solves only for position:

```
Δq = α · J⁺ · (p_desired - p_current)
```

Where:
- `α`: step size
- `J⁺`: damped pseudoinverse of Jacobian
- `Δq`: change in joint angles

Damped pseudoinverse:

```
J⁺ = Jᵀ · (J·Jᵀ + λ²·I)⁻¹
```

---

## 🧠 Notes

- Orientation IK requires working with SE(3) Lie algebra (twist vector).
- Starting with position-only IK simplifies debugging.

---

## 🎯 5. Target Pose Execution

To compute joint angles that reach a desired position:

```python
# Define desired pose (position only)
T_goal = sm.SE3(0.5, 0.2, 0.9)
pos_goal = T_goal.t

# Solve IK
theta_sol = inverse_kinematics_position_only(pos_goal, q0)

# Evaluate
T_result = forward_kinematics(theta_sol, d, a, alpha)[-1]
print("Final position:", T_result[:3, 3])
```

---
