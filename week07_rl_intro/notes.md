# Lecture 6: Introduction to Reinforcement Learning

**Instructor:** Nikolay Karpachev
**Course:** ML MIPT Advanced
**Date:** 11.03.2024

---

## 🧭 Outline

1. Reinforcement Learning (RL) problem statement
2. Multi-Armed Bandits
3. Markov Decision Process (MDP) formalism
4. Relations to Psychology
5. Cross-Entropy Method
6. Comparison: RL vs Supervised vs Unsupervised Learning

---

## 🎯 Reinforcement Learning Problem Statement

### Supervised Learning Recap

**Given:**

- Objects $x_i$ and reference answers $y_i$
- Loss/objective function $\mathcal{L}(y_i, \hat{y_i})$
- Model family $f_\theta$

**Goal:**
Find optimal mapping $f_\theta(x_i) \to y_i$ minimizing loss.
Assumptions: differentiable loss, i.i.d. samples, labeled data.

---

### Reinforcement Learning

**Given:**

- Environment with states $s_t$
- Possible actions $a_t$
- Reward signal $r_t$ (feedback)
- Policy $\pi(a|s)$ — maps states to actions

**Goal:**
Find a policy $\pi$ maximizing **long-term reward** — usually _non-differentiable_ and with _no explicit labels_.

Example:
Train a robot to walk → reward = distance walked.

---

### Components of RL

- **State ($s_t$):** observation or environment description
- **Action ($a_t$):** decision made by the agent
- **Reward ($r_t$):** scalar feedback signal
- **Policy ($\pi$):** mapping from state to action
- **Environment:** defines state transitions and reward rules

$$
s_{t+1}, r_t \sim P(\cdot | s_t, a_t)
$$

---

## 🎰 Multi-Armed Bandits

Simplest RL setup — **no state transitions**, just repeated actions.

- Agent chooses an arm $a_t$ (action)
- Receives stochastic reward $r_t$

Goal:

$$
\max_a ; \mathbb{E}[r_t | a]
$$

Challenge: **exploration vs. exploitation**

- _Exploit_: choose best-known arm
- _Explore_: try new arms to improve knowledge

---

## 🔄 Markov Decision Process (MDP)

Formal definition:

$$
\mathcal{M} = (S, A, P, R, \gamma)
$$

- $S$: set of states
- $A$: set of actions
- $P(s'|s,a)$: transition probabilities
- $R(s,a)$: expected reward
- $\gamma \in [0,1)$: discount factor

### Markov Property

The next state depends only on the current one:

$$
P(s_{t+1}|s_t, a_t, s_{t-1}, a_{t-1}, \dots) = P(s_{t+1}|s_t, a_t)
$$

---

### Total Return

The **cumulative reward** for one episode:

$$
R_{\text{total}} = \sum_{t=0}^{T} \gamma^t r_t
$$

Objective:

$$
J(\pi) = \mathbb{E}*\pi [R*{\text{total}}]
$$

Maximize expected return by improving policy $\pi(a|s)$.

---

## 🧠 Psychology Origins

Early behaviorist experiments inspired RL:

- **Thorndike’s Puzzle Box (1898)** — cats learning escape behavior.
- **Pavlov’s Classical Conditioning** — stimulus-response pairing.
- **Skinner Box (Operant Conditioning)** — behavior shaped by rewards/punishments.

These form the conceptual foundation for _trial-and-error learning_.

---

## ⚙️ How to Maximize Reward

Iterative improvement loop:

1. Run several episodes with current policy $\pi$
2. Collect trajectories $(s_t, a_t, r_t)$
3. Update $\pi$ using feedback
4. Repeat until convergence

---

## 🎯 Cross-Entropy Method (CEM)

A simple yet powerful **policy optimization** algorithm.

---

### 1. Tabular Case

Policy represented as a probability table $\pi(a|s)$.

**Algorithm:**

1. Initialize $\pi$ randomly (each row sums to 1).
2. Sample $N$ trajectories using $\pi$.
3. Select top $M$ _elite_ trajectories with highest rewards.
4. Update $\pi$ based on elite state–action pairs.
5. Repeat until convergence.

---

### 2. Approximate (Parametric) Version

When the number of states is large/infinite.

- Replace tabular policy with parametric model $\pi_\theta(a|s)$.
- Train a model to predict _actions_ from _states_ in elite sessions.

$$
\max_\theta ; \sum_{(s,a) \in \text{elite}} \log \pi_\theta(a|s)
$$

Can use:

- Classification models (for discrete actions)
- Regression models (for continuous actions)

Example:

```python
model.fit(elite_states, elite_actions)
```

---

### 3. Practical Tips

- Use elites from multiple past iterations → stabilize learning.
- Add **entropy regularization** to encourage exploration:
  $$ \mathcal{L}\_{\text{entropy}} = -\lambda \sum_a \pi(a|s) \log \pi(a|s) $$
- Run sessions in parallel.
- Add recurrent memory (RNNs) for partially observable tasks.

---

## 🧩 RL vs. Other Learning Paradigms

| Aspect              | Supervised                         | Unsupervised          | Reinforcement              |
| ------------------- | ---------------------------------- | --------------------- | -------------------------- |
| **Goal**            | Learn mapping from input to target | Find hidden structure | Learn optimal strategy     |
| **Feedback**        | Labeled data (targets)             | None                  | Reward signal              |
| **Data dependency** | i.i.d. samples                     | i.i.d. samples        | Sequential dependence      |
| **Effect on data**  | Model doesn’t affect inputs        | No effect             | Actions change environment |

---

## 🧠 Key Takeaways

- RL learns **through interaction** — feedback, not labels.
- Reward design is critical — affects behavior strongly.
- MDP formalism provides the mathematical backbone.
- **Cross-Entropy Method** is a simple yet effective way to train policies.
- RL lies between supervised and unsupervised learning but introduces _agency_ and _dynamics_.

---

### 📚 References

- DeepMind: _Emergence of Locomotion Behaviors in Rich Environments_
  [YouTube video](https://youtu.be/hx_bgoTF7bs)
- Andrew Ng’s PhD Thesis: _Shaping and Policy Search in Reinforcement Learning_
- Thorndike (1898), Pavlov (1927), Skinner (1953)
- YSDA Practical RL course, UC Berkeley CS188

---

# Lecture 7: Model-Based Reinforcement Learning

**Instructor:** Nikolay Karpachev
**Course:** ML MIPT Advanced
**Date:** 18.03.2024

---

## 🧭 Lecture Overview

1. MDP formalism
2. Improving the Cross-Entropy Method (CEM)
3. Reward design and discounting
4. State-value and action-value functions
5. Bellman equations (expectation & optimality)
6. Policy evaluation and improvement
7. Generalized Policy Iteration (GPI), Policy Iteration, and Value Iteration

---

## 🔁 Recap: From CEM to Model-Based RL

**Cross-Entropy Method (CEM):**

- Easy to implement and effective
- Theoretically grounded
- **Black box:**

  - No explicit knowledge of environment dynamics
  - No intermediate reward modeling

**Model-based RL goal:**
→ Open the black box — use environment **dynamics** to optimize policies more efficiently.

---

## 🧩 The MDP Formalism

An MDP is defined as:

$$
\mathcal{M} = (S, A, P, R, \gamma)
$$

Where:

- $S$: set of states
- $A$: set of actions
- $P(s'|s,a)$: transition dynamics
- $R(s,a)$: reward function
- $\gamma$: discount factor

**Goal:** Find a policy $\pi(a|s)$ that maximizes the expected cumulative reward.

---

## 🎯 The RL Objective

### Reward Hypothesis (Sutton, 1998)

> “All goals and purposes can be described by the maximization of the expected cumulative reward.”

The **return** is the cumulative discounted reward:

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
$$

Objective:

$$
J(\pi) = \mathbb{E}_\pi [G_t]
$$

---

## ⚖️ Reward Design

### Examples of Poor Reward Design

#### Example 1: Data center cooling

- States: temperature readings
- Actions: fan speeds
- Reward:

  - $R = 0$ if overheating
  - $R = +1$ per second system stays cool

**Problem:** infinite return for suboptimal but “safe” behavior.

---

#### Example 2: Navigation task

- State: position and velocity
- Reward:
  $$R = \max(0, d(x, B) - d(x', B))$$
  where $B$ is the goal position.

**Problem:** positive feedback loop — agent oscillates near goal.

---

### Key Takeaways

- Reward **for what**, not **how**.
- Don’t shift or subtract mean rewards arbitrarily — can distort optimization.

---

### Faulty Reward Function Examples

- Reward for ball possession → agent vibrates near ball (soccer task).
- Cyclic behaviors from misaligned shaping.

---

## 💸 Reward Discounting

To make infinite reward sums finite, use discount factor $\gamma$:

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
$$

- $\gamma \in [0, 1)$ controls the value of future rewards.
- Lower $\gamma$ → more short-term focus.
- Typical values: $\gamma = 0.9, 0.95, 0.99$
  → corresponds to effective horizons of ≈ 10, 20, and 100 steps.

---

### Human Discounting

Humans exhibit **hyperbolic** and **quasi-hyperbolic** discounting (Laibson, 1997).
Mathematically convenient exponential discounting assumes a _stationary end-of-effect model_.

---

### End-of-Effect Model

Each action affects:

1. Immediate reward
2. Next state → indirectly affects future rewards

The “effect continuation probability” $\gamma$ controls how far this influence propagates.

---

## ⚙️ Reward Transformations

### Policy-Invariant Transformations (Ng et al., 1999)

#### 1. Reward Scaling

Dividing all rewards by a positive constant:

$$
R'(s,a,s') = c \cdot R(s,a,s')
$$

does **not** change the optimal policy.

#### 2. Potential-Based Reward Shaping

$$
R'(s,a,s') = R(s,a,s') + \gamma F(s') - F(s)
$$

Intuition: adds potential difference between states; preserves optimal policy.

---

## 🧠 Value Functions

### State-Value Function

$$
v_\pi(s) = \mathbb{E}_\pi [G_t | S_t = s]
$$

→ Expected return starting from $s$, following $\pi$.

### Action-Value Function

$$
q_\pi(s,a) = \mathbb{E}_\pi [G_t | S_t = s, A_t = a]
$$

→ Expected return starting from $s$, taking action $a$, then following $\pi$.

---

### Relationship Between $v_\pi$ and $q_\pi$

$$
v_\pi(s) = \sum_a \pi(a|s) q_\pi(s,a)
$$

$$
q_\pi(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) v_\pi(s')
$$

---

## 🔁 Bellman Equations

### Bellman Expectation Equations

For value and action-value functions:

$$
v_\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma v_\pi(s')]
$$

$$
q_\pi(s,a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') q_\pi(s',a')]
$$

---

### Bellman Optimality Equations

For optimal policy $\pi^*$:

$$
v^*(s) = \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma v^*(s')]
$$

$$
q^*(s,a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma \max_{a'} q^*(s',a')]
$$

A **deterministic optimal policy** always exists:

$$
\pi^*(s) = \arg\max_a q^*(s,a)
$$

---

## 🧮 Policy Evaluation and Improvement

### Policy Evaluation

Compute $v_\pi(s)$ for a fixed policy $\pi$ using Bellman expectation equations.

→ System of linear equations:
#states = #unknowns.

---

### Policy Improvement

Given $v_\pi(s)$, create an improved policy:

$$
\pi'(s) = \arg\max_a q_\pi(s,a)
$$

**Policy Improvement Theorem:**
If $\pi'(s)$ is greedy w.r.t. $v_\pi(s)$, then:

$$
v_{\pi'}(s) \ge v_\pi(s) \quad \forall s
$$

Convergence:
If $\pi' = \pi$, policy is optimal.

---

## 🔄 Generalized Policy Iteration (GPI)

Iterative combination of evaluation and improvement:

**Algorithm:**

1. Evaluate policy → estimate $v_\pi(s)$
2. Improve policy → make it greedy w.r.t $v_\pi(s)$

### Variants

| Method                    | Evaluation                  | Improvement    |
| ------------------------- | --------------------------- | -------------- |
| **Policy Iteration (PI)** | Until convergence           | Once per loop  |
| **Value Iteration (VI)**  | One update (truncated eval) | Each iteration |

---

### Complexity Comparison

| Method               | Per Iteration Cost | Iterations | Notes |     |      |      |                            |       |                             |
| -------------------- | ------------------ | ---------- | ----- | --- | ---- | ---- | -------------------------- | ----- | --------------------------- |
| **Value Iteration**  | $O(                | A          |       | S   | ^2)$ | Many | Faster updates, more steps |       |                             |
| **Policy Iteration** | $O(                | A          |       | S   | ^2 + | S    | ^3)$                       | Fewer | Slower updates, fewer steps |

Trade-off → tune number of evaluation steps for best efficiency.

---

## 🧠 Summary

| Concept                           | Description                                        |
| --------------------------------- | -------------------------------------------------- |
| **MDP**                           | Formal structure for RL problems                   |
| **Reward Design**                 | Define _what_ to optimize, not _how_               |
| **Discounting**                   | Finite expected return, balances future vs present |
| **Value Functions**               | Estimate long-term returns                         |
| **Bellman Equations**             | Recursive value definitions                        |
| **Policy Evaluation/Improvement** | Foundation for optimal control                     |
| **GPI**                           | Combines evaluation and improvement iteratively    |

---

**Recommended Reading**

- Sutton & Barto, _Reinforcement Learning: An Introduction_
- Ng et al. (1999), _Policy Invariance under Reward Transformations_
- Laibson (1997), _Golden Eggs and Hyperbolic Discounting_
