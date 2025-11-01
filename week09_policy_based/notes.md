# Lecture 10: Policy Gradient Methods

**Instructor:** Nikolay Karpachev
**Course:** ML MIPT Advanced
**Date:** 8.04.2024

---

## 🧭 Overview

1. Value-based vs Policy-based RL
2. Policy Gradient methods
3. Monte-Carlo (REINFORCE)
4. Actor-Critic algorithms
5. Advantage Actor-Critic (A2C)

---

## ⚖️ Value-based vs Policy-based RL

### Value-Based Methods

- Learn value function ($V$, $Q$).
- Derive implicit policy (e.g., $\varepsilon$-greedy).
- Example: DQN.

### Policy-Based Methods

- Learn the policy directly $\pi_\theta(a|s)$.
- No explicit value function.
- Can represent **stochastic** or **continuous** actions.

### Actor-Critic (Hybrid)

- Learn both:

  - Policy (Actor)
  - Value function (Critic)

---

## 📈 Policy-Based Reinforcement Learning

### Advantages

✅ Smooth policy updates (better convergence)
✅ Naturally handles continuous actions
✅ Learns stochastic behavior

### Disadvantages

❌ Can converge to local optima
❌ High variance gradient estimates

---

## 🎯 Policy Gradient Objective

Given a parameterized policy $\pi_\theta(a|s)$, the goal is to maximize expected return:

$$
J(\theta) = \mathbb{E}*{\tau \sim \pi*\theta}[R(\tau)]
$$

where $\tau = (s_0, a_0, s_1, \dots)$ is a trajectory.

### Optimization

We perform **gradient ascent**:

$$
\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)
$$

---

## ⚙️ Computing the Policy Gradient

Exact gradient computation is difficult because:

- The stationary state distribution $p_\pi(s)$ is unknown.
- Continuous action spaces are intractable.
- Expectations over all trajectories are computationally expensive.

---

## 💡 Policy Gradient Approximations

### 1. **Finite Differences (Numerical)**

Approximate the gradient by evaluating $J(\theta)$ along each parameter axis:

$$
\nabla_\theta J(\theta_i) \approx \frac{J(\theta_i + \varepsilon) - J(\theta_i - \varepsilon)}{2\varepsilon}
$$

**Pros:**

- Works for non-differentiable policies

**Cons:**

- Noisy, slow ($O(n)$ evaluations for $n$ parameters)

---

### 2. **Monte Carlo Estimation**

Express the objective as an expectation over trajectories:

$$
J(\theta) = \mathbb{E}*{\tau \sim \pi*\theta}[R(\tau)]
$$

Use **log-derivative trick** (a.k.a. REINFORCE identity):

$$
\nabla_\theta J(\theta) = \mathbb{E}*{\tau \sim \pi*\theta}!\left[ \nabla_\theta \log P_\theta(\tau) , R(\tau) \right]
$$

Since $P_\theta(\tau) = \prod_t \pi_\theta(a_t|s_t) P(s_{t+1}|s_t,a_t)$,
the environment dynamics cancel out, leaving:

$$
\nabla_\theta J(\theta) = \mathbb{E}*{\pi*\theta}!\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) , G_t \right]
$$

where $G_t$ is the return from step $t$.

---

## 🧮 Monte Carlo Policy Gradient (REINFORCE)

**Algorithm:**

1. Sample $N$ episodes using current policy $\pi_\theta$
2. For each step $(s_t, a_t, r_t)$ compute return
   $$G_t = \sum_{k=t}^T \gamma^{k-t} r_k$$
3. Compute gradient estimate:
   $$
   g = \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) G_t
   $$
4. Update parameters:
   $$
   \theta \leftarrow \theta + \alpha g
   $$

---

### Issues with REINFORCE

- High variance (since gradient depends on full episode returns).
- Unstable convergence.
- Each update uses Monte Carlo rollouts (slow).

---

## 🧠 Actor–Critic Methods

### Idea

Introduce a **critic** (value function) to estimate expected returns instead of Monte Carlo sampling.

Two components:

1. **Actor** – parameterized policy $\pi_\theta(a|s)$
2. **Critic** – parameterized value function $V_w(s)$ or $Q_w(s,a)$

---

### Training Loop

1. **Critic Update:**
   Learn $V_w(s)$ using TD, TD($\lambda$), or MC methods:
   $$
   w \leftarrow w - \alpha_c \nabla_w (V_w(s_t) - \hat{R}_t)^2
   $$
2. **Actor Update:**
   Update policy using critic’s estimate:
   $$
   \theta \leftarrow \theta + \alpha_a \nabla_\theta \log \pi_\theta(a_t|s_t) , Q_w(s_t, a_t)
   $$

The critic stabilizes learning by reducing variance of return estimates.

---

## ⚖️ Advantage Actor–Critic (A2C)

### Motivation

Further reduce variance with a **baseline**.

Any function $B(s)$ independent of action can be subtracted from the return:

$$
\nabla_\theta J(\theta) = \mathbb{E}!\left[\nabla_\theta \log \pi_\theta(a_t|s_t) , (Q(s_t, a_t) - B(s_t))\right]
$$

Choosing $B(s_t) = V(s_t)$ gives the **advantage function**:

$$
A(s_t, a_t) = Q(s_t, a_t) - V(s_t)
$$

Thus, the gradient becomes:

$$
\nabla_\theta J(\theta) = \mathbb{E}!\left[\nabla_\theta \log \pi_\theta(a_t|s_t) , A(s_t, a_t)\right]
$$

### Benefits

- Baseline does not change the expectation (unbiased)
- Reduces variance → more stable learning

---

## 🧩 Summary

| Method                           | Description                      | Pros                     | Cons                         |               |
| -------------------------------- | -------------------------------- | ------------------------ | ---------------------------- | ------------- |
| **Value-Based (e.g. DQN)**       | Learn $Q(s,a)$                   | Stable, sample-efficient | Hard with continuous actions |               |
| **Policy-Based (REINFORCE)**     | Directly optimize $\pi\_\theta(a | s)$                      | Handles continuous actions   | High variance |
| **Actor–Critic**                 | Combines policy & value learning | Lower variance           | More complex                 |               |
| **A2C (Advantage Actor–Critic)** | Uses advantage function          | Stable & efficient       | Requires good critic         |               |

---

## 📚 References

- David Silver, _Reinforcement Learning Lectures_ ([link](https://www.davidsilver.uk/teaching/))
- Yandex Data School, _Practical RL Course_
- Williams, R. J. (1992). _Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning_ (REINFORCE)
