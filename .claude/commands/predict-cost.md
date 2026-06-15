Run the synaxi-predict cost estimator on the task description provided.

Capture the current working directory first, then execute:

```bash
REPO_PATH="$(pwd)" && cd /Users/brad/Code/synaxi-predict && source .venv/bin/activate && python -m predictor.predict "$ARGUMENTS" --models single --repo-path "$REPO_PATH"
```

After showing the output, give a one-sentence interpretation of the recommendation — specifically whether the cost difference justifies using the more expensive model, given the pass rate gap shown.
