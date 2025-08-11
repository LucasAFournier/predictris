# Algorithm: Agent Learning and Prediction

We outline the pseudocode for the agent's learning process, focusing on how it builds and utilizes a forest of prediction trees to understand its environment.

**Data Structures:**

*   `PredictionForest (F)`: A collection of `PredictionTree`s, indexed by the observation they predict.
*   `ActiveSequences (S)`: A mapping from each `PredictionTree` in `F` to a set of `ActiveSequence` objects currently being tracked within that tree.
*   `ActiveSequence`: A tuple `(currentNode, entryNode, context)` representing a path through a `PredictionTree`.
*   `Context`: A tuple `(previousObservation, action)` that led to the activation of a sequence at its `entryNode`.

---

### **Algorithm 1: Agent Learning Step**

**Requires:** Agent's internal state: `PredictionForest F`, `ActiveSequences S`, `previousObservation o_prev`.

1.  **Action Selection:**
2.  `a` ← Select a random action from the set of all possible actions.
3.  Execute action `a` in the environment.
4.  
5.  **Context Update:**
6.  `C_current` ← `(o_prev, a)`
7.  
8.  **Perception:**
9.  `o_curr` ← Perceive the new observation from the environment.
10. 
11. **Tree Creation:**
12. **If** no `PredictionTree` exists for `o_curr` in `F`:
13. &nbsp;&nbsp;&nbsp;&nbsp;`T_new` ← Create a new `PredictionTree` for `o_curr`.
14. &nbsp;&nbsp;&nbsp;&nbsp;Add `T_new` to `F`.
15. 
16. **Context Propagation:**
17. `S_new` ← an empty mapping for new active sequences.
18. **For each** `PredictionTree T` in `F`:
19. &nbsp;&nbsp;&nbsp;&nbsp;`S_new[T]` ← **UpdateSequencesForTree**(`T`, `o_curr`, `a`, `S[T]`, `C_current`)
20. 
21. **State Update:**
22. `S` ← `S_new`
23. `o_prev` ← `o_curr`

---

### **Algorithm 2: UpdateSequencesForTree**

**Requires:** A `PredictionTree T`, current observation `o_curr`, action `a`, the set of active sequences for the tree `S_T`, and the current context `C_current`.
**Returns:** The updated set of active sequences for the tree.

1.  `M` ← Set of all nodes in `T` that match observation `o_curr`.
2.  `S'_updated` ← an empty set.
3.  
4.  **For each** active sequence `s` in `S_T`:
5.  &nbsp;&nbsp;&nbsp;&nbsp;`(n_next, a_edge)` ← Get the successor node and action from the edge leaving `s.currentNode`.
6.  &nbsp;&nbsp;&nbsp;&nbsp;**If** `a_edge` = `a`:
7.  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**If** `n_next` is the designated prediction node of `T`:
8.  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`isCorrect` ← (`o_curr` matches the observation predicted by `T`).
9.  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**UpdatePrediction**(`T`, `s.entryNode`, `s.context`, `isCorrect`).
10. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Else if** `n_next` is in `M`:
11. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`s.currentNode` ← `n_next`.
12. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Add `s` to `S'_updated`.
13. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Remove `n_next` from `M`.
14. 
15. **For each** remaining node `n_match` in `M`:
16. &nbsp;&nbsp;&nbsp;&nbsp;`s_new` ← Create a new `ActiveSequence(n_match, n_match, C_current)`.
17. &nbsp;&nbsp;&nbsp;&nbsp;Add `s_new` to `S'_updated`.
18. 
19. **Return** `S'_updated`.

---

### **Algorithm 3: UpdatePrediction**

**Requires:** A `PredictionTree T`, an `entryNode`, a `context`, and a boolean `isCorrect`.

1.  `n_entry` ← Get the node data for `entryNode` from `T`.
2.  **If** `isCorrect` is `False`:
3.  &nbsp;&nbsp;&nbsp;&nbsp;// The prediction was wrong, so this path is no longer considered reliable.
4.  &nbsp;&nbsp;&nbsp;&nbsp;Set `n_entry.confident` to `False`.
5.  **Else if** `n_entry.confident` is `False`:
6.  &nbsp;&nbsp;&nbsp;&nbsp;// The prediction was right although this path was not considered reliable.
7.  &nbsp;&nbsp;&nbsp;&nbsp;`o_context`, `a_context` ← `context`.
8.  &nbsp;&nbsp;&nbsp;&nbsp;// Reinforce the tree by adding the context that led to the correct prediction.
9.  &nbsp;&nbsp;&nbsp;&nbsp;**If** a path `(o_context) --a_context--> (n_entry)` does not already exist in `T`:
10.  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`n_new` ← Create a new node in `T` representing `o_context`.
11. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Add a directed edge from `n_new` to `n_entry`, labeled with action `a_context`.
12. &nbsp;&nbsp;&nbsp;&nbsp;// The path has been reinforced and is now considered confident.
13. &nbsp;&nbsp;&nbsp;&nbsp;Set `n_entry.confident` to `True`.