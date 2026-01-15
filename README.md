# Multi-Agent Reinforcement Learning for Traffic Signal Control 🚦

This repository implements a reinforcement learning–based traffic signal control system,
progressing from a **single isolated intersection** to a **coordinated multi-intersection road network**
using **multi-agent reinforcement learning (MARL)**.

The project focuses on understanding how **realistic traffic constraints** and **inter-agent coordination**
affect traffic optimization performance.

---

## 📌 Key Features

- Reinforcement learning–based traffic signal control (PPO)
- Realistic traffic constraints:
  - Waiting time modeling
  - Signal switching penalties
  - Minimum green time enforcement
- Single-intersection control (baseline & analysis)
- Multi-intersection (2×2 grid) coordinated control
- Parameter-sharing multi-agent RL
- Direction-aware downstream congestion modeling (spillback awareness)

---

## 🧠 System Overview

### Single Intersection
- One intersection with four incoming lanes (N, S, E, W)
- Agent controls North–South or East–West green phase
- Used to study reward design and constraint effects

### Multi-Intersection (2×2 Grid)

0 ── 1
|    |
2 ── 3

- Four intersections, each controlled by a shared PPO policy
- Vehicles move between intersections
- Global reward encourages coordination
- Downstream pressure enables anticipatory control

---

## 📂 Project Structure

envs/
├── single_intersection_env.py
├── multi_intersection_env.py
train_single.py
train_multi.py
test_single.py
test_multi.py


---

## 🧪 Environments

### Single Intersection (`SimpleTrafficEnv`)
**State:**
[qN, qS, qE, qW, wN, wS, wE, wW]

**Actions:**
0 → North–South green
1 → East–West green

---

### Multi-Intersection (`MultiIntersectionEnv`)
**State per intersection:**
[qN, qS, qE, qW,
wN, wS, wE, wW,
downstream_pressure]

- Downstream pressure captures congestion in receiving lanes of neighboring intersections
- Enables coordination and spillback avoidance

---

## 🎯 Reward Function

Global reward shared by all agents:
reward = - (total queue length + α × total waiting time) - switch penalties

Encourages:
- Reduced congestion
- Fairness (low waiting time)
- Network-level coordination

---

## 🚀 Training

### Single Intersection
```bash
python train_single.py
Multi-Intersection
python train_multi.py
👀 Evaluation
python test_single.py
python test_multi.py
These scripts run trained policies in deterministic mode and print per-intersection traffic states.
