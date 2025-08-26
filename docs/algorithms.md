## Algorithms

We summarise the online learning and update procedures.

### Data Structures

*   `PredictionForest (F)`: collection of `PredictionTree`s, indexed by the observation they predict.
*   `Sequences (S)`: mapping from each `PredictionTree` to its set of active `Sequence`s.
*   `Sequence`: tuple `(current_node, source_node, context)` representing progression through a tree.
*   `Context`: tuple `(previousObservation, action)` that led to activation at a source node.

The main learning loop (Algorithm 1) orchestrates action selection, perception, and memory updates at each timestep.

```python
"""
Requires:
  PredictionForest F
  Sequences S
  previous observation o_prev
"""
1.  # Action Selection
2.  a ← Select a random action from the set of all possible actions.
3.  Execute action a in the environment.
4.  
5.  # Context Update
6.  C_current ← (o_prev, a)
7.  
8.  # Perception
9.  o_curr ← Perceive the new observation from the environment.
10. 
11. # Tree Creation
12. if no PredictionTree exists for o_curr in F:
13.     T_new ← Create a new PredictionTree for o_curr.
14.     Add T_new to F.
15. 
16. # Context Propagation
17. S_new ← an empty mapping for new active sequences.
18. for each PredictionTree T in F:
19.     S_new[T] ← UpdateSequencesForTree(T, o_curr, a, S[T], C_current)
20. 
21. # State Update
22. S ← S_new
23. o_prev ← o_curr
```

**Algorithm 1:** Agent Learning Step

```python
"""
Requires:
  PredictionTree T
  current observation o_curr
  last action a
  set of active sequences for the tree S_T
  current context C_current
Returns:
  updated set of active sequences for the tree
"""
1.  M ← Set of all nodes in T that match observation o_curr.
2.  S_updated ← an empty set.
3.  
4.  # Propagate existing sequences
5.  for each active sequence s in S_T:
6.      (n_next, a_edge) ← Get successor and action from edge leaving s.current_node.
7.      if a_edge == a:
8.          if n_next is the prediction node of T:
9.              # Prediction reached: update confidence based on correctness
10.             is_correct ← (o_curr matches the observation predicted by T).
11.             UpdatePrediction(T, s.source_node, s.context, is_correct).
12.         else if n_next is in M:
13.             # Continue sequence along the matching path
14.             s.current_node ← n_next.
15.             Add s to S_updated.
16.             Remove n_next from M. # Node is now part of a propagated sequence
17. 
18. # Activate new sequences on remaining matching nodes
19. for each remaining node n_match in M:
20.     s_new ← Create a new Sequence(n_match, n_match, C_current).
21.     Add s_new to S_updated.
22. 
23. return S_updated
```

**Algorithm 2:** UpdateSequencesForTree

```python
"""
Requires:
  PredictionTree T
  source_node
  context
  boolean is_correct
"""
1.  if is_correct is False:
2.      # Prediction was wrong, so this path is no longer reliable.
3.      source_node.confident ← False
4.  else if source_node.confident is False:
5.      # Correct prediction from an uncertain path: reinforce by extending it.
6.      o_context, a_context ← context
7.      
8.      # Check if the reinforcing path already exists
9.      if a path (o_context) --a_context--> (source_node) does not already exist in T:
10.         # Create a new node and edge to represent the context.
11.         n_new ← Create a new node in T representing o_context.
12.         Add a directed edge from n_new to source_node, labeled with a_context.
13.     
14.     # The original path is now considered confident.
15.     source_node.confident ← True
```

**Algorithm 3:** UpdatePrediction
