
# Inverse Kinematics: Moore–Penrose Pseudo-Inverse vs Damped Least Squares (DLS)

This repository demonstrates and compares two common inverse kinematics (IK) solvers:

- **Moore–Penrose Pseudo-Inverse (Pinv)**
- **Damped Least Squares (DLS)**, also known as Levenberg–Marquardt

---

## Overview

Inverse kinematics aims to find joint angles changes (Δq) that reduce the task-space error (e) between the current and target end-effector poses.

The Jacobian matrix **J(q)** maps joint velocities to end-effector velocities:

\[
J(q) \Delta q = e
\]

Where:

- \( J(q) \) is the Jacobian at current joint configuration \( q \)
- \( \Delta q \) is the joint angle increment to compute
- \( e \) is the task-space error vector (e.g., position error)

---

## 1. Moore–Penrose Pseudo-Inverse (Pinv)

For non-square Jacobians, the pseudo-inverse provides a least-squares solution:

\[
\Delta q = J^{+} e = J^T (J J^T)^{-1} e
\]

### Problem:

- When \( J J^T \) is near singular (determinant close to zero), inversion is numerically unstable.
- This leads to **large, erratic joint motions** near singularities (robot configurations where Jacobian loses rank).
- For example, small changes in task-space can cause huge joint jumps, risking hardware damage or poor convergence.

---

## 2. Damped Least Squares (DLS)

To stabilize near singularities, DLS adds a damping factor \( \lambda > 0 \) to the normal equations:

\[
\Delta q = J^T (J J^T + \lambda^2 I)^{-1} e
\]

Where:

- \( I \) is the identity matrix
- \( \lambda \) is a small positive damping factor (e.g., 0.01–0.1)

### Benefits:

- Ensures \( J J^T + \lambda^2 I \) is always invertible.
- Sacrifices exactness for smoothness and stability.
- Penalizes large joint updates, preventing erratic motions near singularities.
- Slower but safer convergence.

---

## 3. Measuring Closeness to Singularity

- The **minimum singular value** \( \sigma_{min} \) of \( J \) indicates closeness to singularity:
  - \( \sigma_{min} \approx 0 \) → near singularity.
- The **condition number** \( \kappa = \sigma_{max} / \sigma_{min} \) blows up near singularities.

```python
U, S, Vt = np.linalg.svd(J)
sigma_min = np.min(S)
condition_number = np.max(S) / sigma_min
```

---

## 4. Python Sketch for IK Solver Comparison

The code below compares Pinv and DLS on the same IK problem and plots:

- Task-space error norm \( \| e \| \)
- Joint motion magnitude \( \| \Delta q \| \)
- Minimum singular value \( \sigma_{min} \)

```python
import numpy as np
import matplotlib.pyplot as plt

def ik_solver(q_init, T_goal, method="pinv", max_iter=200, alpha_step=0.5, lam=0.05):
    q = q_init.copy()
    errors, dq_norms, sigmas = [], [], []
    
    for _ in range(max_iter):
        T_all = forward_kinematics(q, robot.d, robot.a, robot.alpha)
        pos_current = T_all[-1][:3, 3]
        error = T_goal - pos_current
        
        errors.append(np.linalg.norm(error))
        
        J = Jacobian(q)[0:3, :]
        
        # SVD for singularity measure
        U, S, Vt = np.linalg.svd(J)
        sigmas.append(np.min(S))
        
        # Choose solver
        if method == "pinv":
            dq = alpha_step * np.linalg.pinv(J) @ error
        elif method == "dls":
            J_damped = J.T @ np.linalg.inv(J @ J.T + (lam**2) * np.eye(3))
            dq = alpha_step * J_damped @ error
        
        dq_norms.append(np.linalg.norm(dq))
        q += dq

        if np.linalg.norm(error) < 1e-4:
            break
    
    return errors, dq_norms, sigmas

# Run both solvers
errors_pinv, dq_pinv, sigma_pinv = ik_solver(q0, np.array([0.5, 0.2, 0.9]), method="pinv")
errors_dls, dq_dls, sigma_dls = ik_solver(q0, np.array([0.5, 0.2, 0.9]), method="dls")

# Plot results
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
```

---

## 5. Interpretation of Plots and Behavior

| Aspect            | Observation                              | Interpretation                                    | Suggested Improvement                      |
|-------------------|----------------------------------------|-------------------------------------------------|--------------------------------------------|
| **Task-space error** (left plot) | Pinv converges faster but oscillates. DLS converges smoothly but slower. | Pinv overshoots near singularities; DLS is stable. | Tune damping \( \lambda \) and step size \( \alpha \) adaptively. |
| **Joint motion norm** (middle plot) | Pinv shows large spikes in joint updates; DLS remains smooth and bounded. | Pinv unstable near singularities; DLS regularizes jumps, safer for hardware. | Limit max joint step size; use adaptive damping increasing near singularities. |
| **Minimum singular value** (right plot) | Initially low indicating closeness to singularity; Pinv dips near iteration 5; DLS stays higher and smoother. | Pinv unstable near singularity; DLS avoids abrupt changes by damping. | Implement adaptive damping based on \( \sigma_{min} \); add null-space optimization for singularity avoidance. |

---

## 6. Summary and Recommendations

- **Moore–Penrose pseudo-inverse (Pinv):** Fast but unstable near singularities, prone to large joint motions.
- **Damped Least Squares (DLS):** More stable, slower convergence, smooth joint motions.
- **Adaptive damping** improves robustness by increasing \( \lambda \) near singularities.
- **Step size control** helps prevent large joint jumps.
- **Singularity avoidance strategies** (null-space optimization, joint limit avoidance) improve IK safety.

---
