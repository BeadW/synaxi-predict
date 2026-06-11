# synaxi-predict — project plan

## What we are building

A system that predicts the cost and optimal execution strategy for a Claude Code agent run before it starts. The prediction is exposed as a hosted API and as a Claude Code skill (`/estimate`). Synaxi uses the same API internally to route transparently.

The system is trained on empirical benchmark data — real agent runs against structured task definitions — rather than heuristics or hand-tuned rules.

---

## Scope (v1)

- Claude Code as the agent harness (Haiku / Sonnet / Opus)
- Synaxi token compression as a first-class strategy option
- Full agent run prediction — total tokens, total cost, recommended strategy
- BDD task definitions as the source of truth for training data
- Hosted API + Claude Code skill as the distribution mechanism

Future scope (not in v1): Bedrock backend, Qwen Coder, other providers.

---

## Strategies to benchmark

We are not just picking one model vs another. We benchmark all viable execution strategies and the predictor recommends among them:

| ID | Name | Description |
|----|------|-------------|
| `single-haiku` | Haiku only | All steps on Haiku. Cheapest, lowest ceiling. |
| `single-sonnet` | Sonnet only | All steps on Sonnet. Default Claude Code behaviour. |
| `single-opus` | Opus only | All steps on Opus. Best quality, highest cost. |
| `advisor` | Advisor strategy | Opus plans + evaluates; Sonnet/Haiku executes. Selective Opus invocation. |
| `cascade` | Cascade routing | Start cheap (Haiku), escalate to Sonnet then Opus on confidence failure or retry. |
| `parallel-vote` | Parallel voting | Run N cheaper models in parallel, majority-vote or Opus-judge the result. Best for high-stakes single-turn decisions. |
| `compressed-sonnet` | Synaxi + Sonnet | Synaxi compression applied before dispatch to Sonnet. |
| `compressed-advisor` | Synaxi + Advisor | Synaxi compression on all steps; Opus advisor, Haiku executor. |
| `speculative` | Speculative draft | Haiku drafts, Opus corrects. Like speculative decoding but at agent level. |

The predictor outputs: estimated cost per strategy, estimated quality score per strategy, and a recommended strategy given the user's cost/quality preference.

---

## Task taxonomy

Tasks are organised into categories and complexity tiers. Each task is a `.feature` file (Gherkin / BDD). This gives us:

- Structured, unambiguous task definitions
- Acceptance criteria that can be auto-evaluated
- A reproducible test suite that grows over time
- A clean feature label for the predictor

### Categories

| Category | Examples |
|----------|---------|
| `code/generation` | Write a function, scaffold a module, implement an algorithm |
| `code/debug` | Find the bug, explain the error, fix the failing test |
| `code/refactor` | Extract method, rename for clarity, reduce duplication |
| `code/review` | Review a diff, flag issues, suggest improvements |
| `research` | Summarise a topic, compare options, answer a factual question |
| `analysis` | Analyse data, identify patterns, produce a report |
| `planning` | Decompose a project, write a spec, estimate effort |
| `writing` | Draft copy, rewrite for tone, proofread |

### Complexity tiers

Each category has three tiers:

- **T1 — simple**: well-bounded, single-step, clear success criterion
- **T2 — moderate**: 2-5 steps, some ambiguity, multiple valid outputs
- **T3 — complex**: open-ended, multi-step, requires planning or sustained context

Target: ~30 feature files per category (10 per tier), ~240 tasks total for v1.

---

## BDD feature file format

Each `.feature` file defines a single task. The structure is:

```gherkin
Feature: [human-readable task name]
  As a developer using Claude Code
  I want to [goal]
  So that [outcome]

  Background:
    Given the following context:
      """
      [any files, code, or setup that must be provided to the agent]
      """

  Scenario: [name of the success case]
    Given an agent is started with the above context
    When the agent completes the task
    Then the output should [acceptance criterion 1]
    And the output should [acceptance criterion 2]

  Scenario: Complexity tier
    Given this task
    Then the complexity is "T2"
    And the category is "code/debug"
```

The harness parses `.feature` files and:
1. Provides the `Background` context to the agent as its initial message
2. Runs the agent to completion under the target strategy
3. Evaluates the `Then` criteria with a judge model
4. Records all token counts, costs, and the judge quality score

---

## Harness architecture

```
features/
  code/
    debug/
      t1_off_by_one.feature
      t2_null_pointer_chain.feature
      ...
  research/
    ...

harness/
  runner.ts          — parse feature files, execute agent runs, record results
  strategies/
    single.ts        — single-model execution
    advisor.ts       — advisor strategy implementation
    cascade.ts       — escalation logic
    parallel_vote.ts — parallel execution + voting
    compressed.ts    — synaxi compression wrapper
    speculative.ts   — speculative draft strategy
  evaluator.ts       — judge model evaluation of Then criteria
  recorder.ts        — write run results to data/runs/

data/
  runs/              — JSONL, one record per agent run
  models/            — trained predictor weights
```

Each run record:

```json
{
  "task_id": "code/debug/t2_null_pointer_chain",
  "strategy": "advisor",
  "complexity": "T2",
  "category": "code/debug",
  "prompt_tokens_raw": 1240,
  "prompt_tokens_compressed": 890,
  "completion_tokens": 3100,
  "total_cost_usd": 0.0041,
  "turn_count": 4,
  "tool_call_count": 6,
  "quality_score": 0.87,
  "passed_criteria": true,
  "timestamp": "2026-06-12T10:00:00Z"
}
```

---

## Predictor model

**Phase 1 — gradient boosted trees (start here)**

Features per task:
- Prompt embedding (small cheap model, e.g. Haiku or a local embedder)
- Heuristic signals: prompt token count, post-compression token count, category label, complexity tier, presence of code blocks, tool list size, imperative verb count
- Synaxi compression ratio for this prompt

Targets (one model per target):
- `cost_p50`, `cost_p90` per strategy
- `quality_score` per strategy
- `recommended_strategy` (classification)

Use XGBoost or LightGBM. Fast to iterate, interpretable feature importances, works well with tabular data at this scale.

**Phase 2 — embedding-based neural net (once we have 10k+ runs)**

Replace heuristic features with a small transformer over the prompt. Fine-tune on run data. Graduate to this when tabular features plateau.

**User-contributed data flywheel:**

With permission, collect actual run data from Synaxi users: task embedding (not raw prompt), strategy used, actual cost, thumbs up/down quality signal. This is the long-term moat. Raw prompts never leave the user's machine — embeddings only.

---

## API

Hosted at `api.synaxi.io/v1/predict` (or similar).

```
POST /v1/predict
{
  "prompt": "...",          // the user's task prompt
  "context_size": 4200,     // current context window tokens
  "tools": ["bash", "read"] // tools available in this session
}

Response:
{
  "strategies": [
    {
      "id": "compressed-advisor",
      "cost_p50_usd": 0.0031,
      "cost_p90_usd": 0.0089,
      "quality_score": 0.91,
      "recommended": true
    },
    ...
  ],
  "task_category": "code/debug",
  "complexity_estimate": "T2"
}
```

The API is free to call. Synaxi users get the strategy applied automatically with no friction.

---

## Claude Code skill

A `/estimate` skill that calls the API and presents the result in the terminal before the user starts a task.

```
/estimate

Analysing task...

  Recommended: Synaxi + Advisor strategy
  Est. cost:   $0.003 (p50) — $0.009 (p90)
  Quality:     91/100

  Alternatives:
    Sonnet only          $0.012   88/100
    Haiku only           $0.004   71/100
    Advisor (no Synaxi)  $0.006   90/100

  → Add Synaxi to reduce cost by ~48% with no quality loss.
```

---

## Build sequence

### Phase 1 — task library (2 weeks)
- Write 240 Gherkin feature files across all categories and tiers
- Define evaluation rubrics for each category (what "passed" means)
- Validate that the harness can parse and execute them end-to-end on a small sample

### Phase 2 — benchmark run (2-3 weeks)
- Run all 240 tasks against all 9 strategies
- That's ~2,160 agent runs
- Parallelize where possible; log everything to `data/runs/`
- Manual spot-check quality scores against judge scores on a 5% sample

### Phase 3 — predictor (1-2 weeks)
- Train XGBoost models on run data
- Validate on held-out 20% split
- Target: recommended strategy correct 75%+ of the time; cost estimate within 2x on p90

### Phase 4 — API + skill (1 week)
- Wrap predictor in a FastAPI service
- Ship the Claude Code skill
- Instrument the API to collect opt-in run data for the flywheel

### Phase 5 — flywheel
- Add user data collection to Synaxi (opt-in, embeddings only)
- Retrain predictor monthly
- Publish accuracy benchmarks publicly (marketing asset)

---

## What makes this defensible

1. Compression-aware cost estimates — we know the post-compression token count; nobody else does
2. Strategy-level prediction, not just model routing — we recommend how to run, not just which model
3. Empirical training data, not heuristics — grounded in real agent runs
4. User data flywheel — opt-in run data compounds the advantage over time
5. Distribution moat — free skill creates top-of-funnel; Synaxi is the obvious upgrade
