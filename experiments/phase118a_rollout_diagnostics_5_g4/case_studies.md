# Phase 1.18a Case Studies

## Case 1: Group with reward spread (good for GRPO)

**Group ID:** `esci_val_2`
**Original query:** expandable outdoor gate
**Rewards:** [-0.1, -0.15000000000000002, -0.1, -0.15000000000000002]
**NDCG@10 values:** [0.0, 0.0, 0.0, 0.0]
**Final queries:** ['expandable outdoor gate', 'expandable outdoor gate 24 foot', 'expandable outdoor gate', 'expandable outdoor gate replacement']
**Trajectory summaries:**
- g0: search('expandable outdoor gate fence') -> search('expandable outdoor gate') -> final('expandable outdoor gate')
- g1: search('expandable outdoor gate 24 foot') -> search('expandable outdoor gate 24 foot') -> final('expandable outdoor gate 24 foot')
- g2: search('expandable outdoor gate replacement') -> search('expandable outdoor gate') -> final('expandable outdoor gate')
- g3: search('expandable outdoor gate replacement') -> search('expandable outdoor gate replacement') -> final('expandable outdoor gate replacement')
**Diagnosis type:** `diverse_trajectory_reward_spread`
**Note:** Trajectories differ and rewards differ. This group can produce meaningful GRPO advantages.
**Diagnosis:** Trajectories and rewards both vary; GRPO advantage should be non-zero.

## Case 2: Diverse trajectories but identical reward

**Group ID:** `esci_val_0`
**Original query:** bathroom fan without light
**Rewards:** [-0.1, -0.1, -0.1, -0.1]
**NDCG@10 values:** [0.0, 0.0, 0.0, 0.0]
**Final queries:** ['bathroom fan without light', 'bathroom fan no light remote', 'bathroom fan without light', 'bathroom fan no light remote']
**Search query sequences:** [['bathroom fan no light', 'bathroom fan without light'], ['bathroom fan no light', 'bathroom fan no light remote'], ['bathroom fan no light', 'bathroom fan without light'], ['bathroom fan no light', 'bathroom fan no light remote']]
**Trajectory summaries:**
- g0: search('bathroom fan no light') -> search('bathroom fan without light') -> final('bathroom fan without light')
- g1: search('bathroom fan no light') -> search('bathroom fan no light remote') -> final('bathroom fan no light remote')
- g2: search('bathroom fan no light') -> search('bathroom fan without light') -> final('bathroom fan without light')
- g3: search('bathroom fan no light') -> search('bathroom fan no light remote') -> final('bathroom fan no light remote')
**Diagnosis type:** `diverse_trajectory_zero_reward`
**Note:** Trajectories differ, but final reward is identical. BM25/NDCG may be insensitive to these query rewrites.
**Why this matters:** Different trajectories do not change BM25/NDCG reward, so GRPO advantage collapses.
**TopK overlap:** avg_pairwise_topk_overlap=0.333

## Case 3: Fully collapsed same-trajectory group

No fully collapsed same-trajectory group found.
