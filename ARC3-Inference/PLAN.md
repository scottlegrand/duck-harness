# ARC-AGI-3 Causal Reasoning Solver Plan

## Objective

Improve the solver by turning its current scientist-style prompting into durable,
machine-enforced causal reasoning. The solver should accumulate evidence about game
mechanics, choose informative experiments while uncertain, switch to programmatic
planning when sufficiently confident, and retain reusable mechanics across levels.

The main limitation is not a lack of additional chain-of-thought instructions. The
current prompt already asks the VLM to form hypotheses, predict outcomes, inspect
transitions, avoid loops, distinguish HUD changes from gameplay changes, and transfer
rules across levels. The missing component is a persistent executable belief state:
today, much of that reasoning remains optional free-form prose that can be overwritten,
trimmed from context, or ignored.

## Target Reasoning Loop

For every uncertain game state, the solver should perform this loop:

1. **Inspect:** derive a compact structural description of the current observation.
2. **Update beliefs:** maintain multiple candidate entity, control, and goal hypotheses.
3. **Predict:** state the expected result of each useful candidate action under those
   hypotheses.
4. **Experiment:** select the safe action with the greatest expected information gain.
5. **Compare:** classify the actual outcome against the prediction and update confidence.
6. **Plan:** once mechanics are reliable, search the inferred symbolic state space with
   Python rather than relying on long mental simulation.
7. **Verify:** treat an environment level transition as the only proof of level success.
8. **Transfer:** retain confirmed causal rules across levels while discarding or
   revalidating level-specific geometry.

## Current Pipeline Gaps

- The working world model is extracted from optional free-form assistant labels and
  stored as one string per category.
- A new statement can overwrite earlier evidence rather than append support or
  contradiction to a hypothesis.
- Useful discoveries emitted only in the model's reasoning channel are not incorporated
  into the structured world-model summary.
- Most accumulated knowledge, including the action model, is cleared when a level
  transition occurs.
- Concepts such as "same effective state," "failed experiment," and "meaningful
  progress" are prompt instructions rather than computed properties.
- The current four-connected, same-color segmentation is useful evidence but is treated
  too strongly as an entity model. It can split multicolored entities, merge touching
  same-colored entities, and confuse large backgrounds with objects.
- Conversational history is not a durable experimental record and can be lost during
  context trimming.
- The system prompt repeats several policies, diluting the operational instructions.
- Default stochastic sampling makes controlled comparisons and regression testing less
  reproducible.

## Proposed Architecture

### 1. Persistent typed causal ledger

Create a ledger owned by the harness and persisted beside runtime state. It should not
depend on chat history or free-form prose.

Initial schema:

```json
{
  "entities": [],
  "control_hypotheses": [
    {
      "id": "control-1",
      "rule": "RIGHT moves entity E1 one logical cell",
      "confidence": 0.8,
      "supporting_transitions": [3, 7],
      "contradicting_transitions": [],
      "status": "active"
    }
  ],
  "goal_hypotheses": [
    {
      "id": "goal-1",
      "rule": "place E1 inside E4",
      "confidence": 0.45,
      "supporting_transitions": [],
      "contradicting_transitions": [],
      "status": "active"
    }
  ],
  "failed_experiments": [],
  "confirmed_cross_level_rules": [],
  "state_signature": "",
  "taboo_state_actions": [],
  "next_experiment": {
    "action": "LEFT",
    "predictions": {},
    "reason": "distinguishes object movement from camera movement"
  }
}
```

Requirements:

- Validate every update against a schema.
- Append evidence rather than replacing the whole record.
- Associate hypotheses with concrete transition IDs.
- Preserve rejected hypotheses so they are not rediscovered repeatedly.
- Distinguish observations, inferences, and confirmed rules.
- Preserve confirmed mechanics across level transitions.
- Reset level-specific entities, geometry, and plans after a transition.

### 2. Deterministic transition analysis

Compute a compact transition record after every real action:

- changed-cell count and bounding regions;
- color-transition counts;
- appearing, disappearing, moving, splitting, and merging components;
- candidate object displacement vectors;
- camera-motion hypotheses;
- edge/HUD-only changes;
- score, level, valid-action, and terminal-state changes;
- animation versus persistent-state changes when multiple frames are available.

The model should receive this derived evidence in addition to the current image and
segmentation. It should not have to reconstruct basic diffs repeatedly through tool
calls.

### 3. Effective-state and loop detection

Calculate an effective-state signature using gameplay-relevant structure, current level,
known modes or selections, and recent causal history when identical frames may represent
different hidden states.

Use it to enforce:

- a state/action taboo table;
- detection of repeated cycles;
- detection of repeated no-effect probes;
- escalation from local replanning to hypothesis replacement;
- optional RESET recommendations without automatically destroying useful state.

The harness should distinguish:

- `no_effect`;
- `hud_or_timer_change`;
- `animation_only`;
- `gameplay_state_change`;
- `progress`;
- `level_complete`.

`board_changed=true` alone must never count as progress.

### 4. Multi-hypothesis perception

Retain current segmentation, but expose it as one interpretation rather than ground
truth. Add derived candidates using:

- four- and eight-connectivity;
- multicolor grouping by enclosure, shared motion, repetition, and proximity;
- several plausible background colors;
- containment and adjacency graphs;
- repeated-shape matching independent of location;
- temporal tracking by overlap, displacement, area, color, and shape;
- separation of edge-aligned HUD regions from the gameplay interior.

The ledger should represent uncertainty when multiple entity decompositions remain
plausible.

### 5. Prediction-versus-outcome enforcement

Before an uncertain action, require a structured experiment declaration:

```json
{
  "hypotheses_tested": ["control-1", "camera-1"],
  "action": {"action": "RIGHT"},
  "predicted_outcomes": {
    "control-1": "E1 moves right by one logical cell",
    "camera-1": "scene shifts left while E1 remains screen-centered"
  },
  "expected_information_gain": 0.7,
  "risk": "low"
}
```

After execution, the harness should attach the observed transition and ask the model to
mark each prediction as supported, contradicted, or inconclusive. Confidence updates
must refer to that evidence.

### 6. Discovery-to-planning transition

Use two explicit modes:

- **Discovery mode:** single-action probes selected for information gain.
- **Planning mode:** programmatic search using mechanics whose confidence exceeds a
  defined threshold.

Planning mode may use BFS, shortest paths, flood fill, constraint solving, permutation
search, or bounded beam search. If a planned transition contradicts the learned forward
model, immediately return to discovery mode rather than extending the failed plan.

Batch actions only when intermediate observations are not needed and every transition in
the batch follows a supported deterministic rule.

### 7. Cross-level knowledge policy

On level completion, persist:

- controllable entities or entity roles;
- action-to-effect mappings;
- collision and interaction rules;
- confirmed goal rule, if reusable;
- successful abstract strategy;
- remaining uncertainty.

Clear or revalidate:

- absolute coordinates;
- paths;
- current object IDs;
- current target assignments;
- level-specific geometry.

Test transferred controls with one low-risk probe on the new level before using them in a
longer plan.

### 8. Prompt reduction

After enforcement exists in code, replace repeated policy prose with a short operational
contract:

```text
Inspect -> update ledger -> predict -> act once -> compare outcome.
When uncertain, maximize information gain.
When confident, search using confirmed mechanics.
Never equate board_changed with progress.
A level transition is the only proof of success.
```

Keep only the runtime API details the model must know. Generate the current ledger,
transition evidence, and required response schema dynamically.

### 9. Reproducibility and evaluation

For development comparisons:

- use a fixed seed;
- reduce sampling variance unless deliberate exploration is being tested;
- keep model, public games, concurrency, timeout, action budget, and prompt version fixed;
- record every ledger mutation and its supporting transition;
- separately measure inference failures and gameplay failures.

Compare variants using:

- aggregate RHAE percentage;
- raw levels solved;
- games reaching each level;
- actions per solved level;
- actions spent in repeated effective states;
- fraction of probes producing information;
- hypothesis prediction accuracy;
- time and model requests per level;
- malformed tool calls and analyzer failures.

## Implementation Phases

### Phase 1: Instrument without changing behavior

- Add transition IDs and deterministic transition summaries.
- Add effective-state signatures and report repeated state/action pairs.
- Persist the existing free-form world model separately from chat history.
- Add metrics for loops, HUD-only changes, and prediction accuracy.
- Confirm that instrumentation does not change actions or scores.

### Phase 2: Introduce the structured ledger

- Define and validate the ledger schema.
- Add Python-tool accessors for reading and updating it.
- Preserve evidence and rejected hypotheses across context trimming.
- Implement explicit level-specific versus cross-level fields.
- Continue including the old free-form summary temporarily for comparison.

### Phase 3: Enforce experimental actions

- Require prediction records for uncertain actions.
- Automatically compare predictions with transition summaries.
- Implement taboo state/action pairs and cycle detection.
- Recommend mechanically different probes after repeated inconclusive actions.

### Phase 4: Add perception alternatives

- Add multiple connectivity and background hypotheses.
- Add temporal grouping and object tracking.
- Add deterministic HUD-region candidates.
- Evaluate whether each added representation improves decisions rather than merely
  increasing context.

### Phase 5: Add planning mode

- Expose a compact symbolic state and supported forward rules.
- Provide reusable bounded-search utilities in the Python environment.
- Gate search on forward-model confidence.
- Fall back to discovery immediately on prediction mismatch.

### Phase 6: Reduce and tune the prompt

- Remove duplicated scientist, anti-loop, and inspection prose now enforced by code.
- Retain a concise operational contract and exact tool semantics.
- Fix seeds for regression runs and evaluate lower-variance sampling settings.
- A/B test against the saved original and current prompts.

## Acceptance Criteria

The work is successful only if it improves gameplay under controlled comparison, not
merely if the implementation runs.

- No regression in tool-call validity or environment action execution.
- Ledger survives context trimming and level transitions according to policy.
- Every uncertain action can be traced to a hypothesis and prediction.
- Repeated no-effect state/action pairs decline substantially.
- The solver detects and stops failed action cycles without relying on model prose.
- Confirmed mechanics are reused on later levels without replaying level-specific paths.
- Public evaluation produces a reproducible non-zero RHAE percentage.
- The structured-ledger variant outperforms the current prompt-only baseline on either
  RHAE or action efficiency without increasing infrastructure failure rate.

## Priority Order

1. Persistent typed causal ledger.
2. Deterministic transition classification and effective-state signatures.
3. Prediction-versus-outcome enforcement.
4. Correct cross-level retention.
5. Multi-hypothesis perception.
6. Discovery/planning mode switch and bounded search utilities.
7. Prompt reduction.
8. Controlled A/B evaluation.

