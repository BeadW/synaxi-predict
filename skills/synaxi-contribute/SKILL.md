---
description: >
  Contribute recorded actuals (task text, turns, cost, pass/fail, tree-sitter
  code features) to the synaxi-predict training dataset via GitHub issue.
when_to_use: >
  NEVER invoke this skill automatically. It must only run when the user
  explicitly types /synaxi-contribute. Do not call it after task completion,
  during recording, or for any other reason.
user-invocable: true
allowed-tools: Bash AskUserQuestion
---

See commands/synaxi-contribute.md for the full execution steps.
This skill is user-invocable only — the model must never invoke it automatically.
