# Predictris

*Developed and maintained by **Lucas Fournier** in collaboration with **Jean-Charles Quinton**, **Mathieu Lefort**, and **Frédéric Armetta***  

Predictris is an open-source framework for **online sensorimotor learning** that lets an autonomous agent discover and represent the regularities of its environment purely by trying to predict what will happen next. It combines **sequence-based memory**, **incremental updates at every time-step**, and a built-in notion of **curiosity** to build compact predictive models without any external rewards.

---

## Overview

Artificial-vision and language models already outperform humans on many benchmarks, yet their skills remain narrow because they lack the **action-grounded structure** that underpins human cognition. Predictris addresses this gap by coupling perception, action, and prediction into a single closed loop, treating perception not as passive input but as an *active* mastery of **sensorimotor contingencies**—systematic relationships between what the agent does and what it then perceives.

---

## Background and Key Principles

* **Sequence-based Representation:** The world model stores traces of the form  
  *(observation *o* , action *a*) → observation *o′* *.*  
  These sequences disambiguate superficially similar observations by their different consequences.

* **Online Learning:** The model refines itself **at every time-step** so the agent can adapt instantly to new contingencies it encounters through interaction.

* **Intrinsic Motivation:** The agent is **curiosity-driven**—its only “reward” is to minimise prediction error. This intrinsic signal compels broad, task-agnostic exploration.

---

## Architecture: Predictive Trees

A **Predictive Tree** is a rooted, labelled digraph that stores the *minimal* history required to forecast a chosen observation:

* **Nodes** hold past observations and a confidence flag.  
* **Edges** are labelled by actions.  
* The **root** is the observation the tree aims to predict.

At each time-step the agent activates every node that matches its current perception and follows the outgoing edge corresponding to the executed action. Reaching the root tests a prediction; failure marks the source node “unconfident”, whereas a later success triggers *backward induction*—a new node for a further-past observation and action is prepended, and the node is relabelled “confident”. Extensions occur **only** on previously unconfident nodes, so each tree converges to the shortest prefix sufficient for reliable prediction, yielding a compact, adaptive memory.

---

## Simulation Environment

We evaluate Predictris in a **Tetris-inspired 2‑D grid world**:

| Element | Description |
|---------|-------------|
| **World** | Discrete grid containing a single tetromino-shaped piece |
| **Sensor** | Agent has a local \(3 × 3\) binary window over the grid |
| **Actions** | Translations (up, down, left, right) and 90° rotations of the piece |
| **Goal (implicit)** | Build predictive trees that anticipate how the \(3 × 3\) view changes under each action |

Because the agent never sees the whole shape at once, it must integrate information over time, and its learned trees ultimately encode the object’s structure.

---

## Features

* **Fully online learning** —no train/eval split; the model updates on every step.  
* **Predictive-tree memory** that grows only when predictions fail, ensuring compactness.  
* **Reward-free exploration** driven purely by intrinsic prediction error.  
* **Configurable exploration** (random vs. prediction-guided) and context-activation policies.  
* **Lightweight environment** for quick prototyping or reproducible experiments.

---

## Citation

If you use Predictris in academic work, please cite:

```bibtex
@inproceedings{fournier2025predictris,
  title     = {Online Sensorimotor Sequence-based Learning Using Predictive Trees},
  author    = {Lucas Fournier and Jean-Charles Quinton and Mathieu Lefort and Fr\'ed\'eric Armetta},
  year      = {2025},
}
```
