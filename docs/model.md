## Model

### Prediction Paths

Drawing on insights from adaptive decision‑tree streams [2, 3], we propose a model using several new objects defined as follows.

The fundamental object we use in our model is called a `PredictionPath`.

A Prediction Path predicts the perceived result of a chosen action in a given situation (Fig. 1). For a vision‑only agent, perception is its next observation . In line with **P1 (Prediction as Intrinsic Motivation)**, prediction supplies the intrinsic training signal: each path is evaluated solely through its predictive success. The prediction is encoded in a `prediction_node` connected to the agent's representation of the situation and asserts: “given this situation, the next observation will be $o$”.

Consistent with **P2 (Sequence‑based Representation for Sensorimotricity)**, the agent's representation of a situation is an observation–action sequence alternating observation nodes and action‑labelled directed edges that lead to the prediction node. We call the first observation node—where the agent recognises the path “begins”—the `source_node`. As observations and actions unfold, paths are traversed during `Sequences`. A sequence starts when the current observation matches the source node, which is marked as the `current_node`. This label propagates along the path until either the path mismatches the interaction or the prediction node becomes the current node.

Following **P3 (Online Learning for Adaptability)**, prediction paths are created incrementally. Intuitively, simple situations yield short paths; ambiguous ones require longer prefixes carrying more recent interaction to support confident prediction. Each path stores a boolean `confidence` state, and each sequence carries a `context`—the (observation, action) pair that occurred just before the sequence’s start at the source node. When a prediction node becomes current, learning proceeds as follows (Algorithm 3)—per **P1**, using predictive correctness as the sole criterion: (i) if an *uncertain* path makes a correct prediction, a new path is created by prepending the sequence’s context to the source node and the original path becomes *confident*; (ii) if the prediction is wrong, the path becomes *uncertain* until further evidence justifies extension.

Extensions therefore arise only on previously uncertain paths, so the paths "converge" to the shortest prefixes sufficient for prediction, yielding a incrementally constructed memory whose depth adapts to the empirical complexity of the environment.

![Example of a prediction path with a sequence (grey dots) activated by a specific observation-action history.](docs/img/prediction_path.svg)

**Figure 1:** Example of a prediction path with a sequence (grey dots) activated by a specific observation-action history.

### Prediction Trees and Forests

Naïvely activating all paths that match the current observation duplicates computation across overlapping prefixes. Concretely, as interaction unfolds the agent asymptotically stores as many prediction paths as there are **unique situations** it has encountered. If a long path $p$ has been memorised, every strict prefix that contributed to building $p$ is also stored, because each prefix corresponded to a past situation that did not yet disambiguate towards $p$. Under naïve activation, when the current interaction matches $p$, the agent simultaneously activates sequences at all matching prefixes. Each prefix then attempts to extend again toward the same prediction node, effectively **reconstructing the long path from scratch** at every step. This yields systematic duplication: many concurrent updates for what is semantically a **single** predictive claim.

We therefore introduce **Prediction Trees** that merge all prediction paths ending in the same predicted observation into a labelled, directed **in‑tree** whose root is the prediction node (Fig. 2). The tree realises two mechanisms that eliminate duplication:

1. **Prefix sharing.** Common prefixes are represented once. Any path from a node to the root constitutes a valid prediction path (**T1**), and a path is uniquely identified by its source node (**T2**). Thus, shorter subpaths do not re‑instantiate separate computations when a longer matching path exists—they are simply the ancestor segment of that longer path inside the tree.
2. **Dominance at activation time.** During updating (Algorithm 2), when a sequence propagates from a node to its child along the executed action, the child is **consumed** and the node is removed from the set of match candidates $M$ (see lines 12–16). This prevents simultaneous activation of both a node and its ancestors, ensuring that only the **deepest matching nodes**—the *frontier*—remain active. Consequently, the long path contributes exactly **one** active sequence; its shorter prefixes are implicitly accounted for by the unique upward route to the root.

Properties of prediction trees:
- **T1.** Any path from a node to the root is a valid prediction path.
- **T2.** Each prediction path is uniquely identified by its source node; path confidence is stored at that node.

The agent maintains one tree per predictable observation; the collection forms a **Prediction Forest**.

### Activation and Confidence‑aware Attention

At each timestep, for each tree, the agent identifies all *matching nodes* whose stored observation equals the current perception $o_t$. It then (i) **propagates existing sequences** along edges that match the executed action $a_t$; (ii) **activates new sequences** on remaining matching nodes (Algorithm 2). Two activation strategies are implemented: the default `'all'` strategy activates all candidate nodes; the `'by_confidence'` strategy activates a node with probability equal to its confidence, focusing computation on historically reliable contingencies.

![Example of the construction of prediction trees for a specific observation–action history with timesteps.](docs/img/prediction_trees.svg)

**Figure 2:** Example of the construction of prediction trees for a specific observation–action history with timesteps. At step $t$ the agent activates every node whose observation matches its current perception $o_t$ (dotted circles) and follows the outgoing edge corresponding to the executed action $a_t$. Reaching the root tests the prediction: a failure flips the source node to *uncertain* (black circle), whereas a subsequent success uses backward induction to append a previous observation–action pair to the start node, after which the node becomes *confident* (plain circle).