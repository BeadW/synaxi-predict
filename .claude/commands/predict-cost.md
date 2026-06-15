Run the synaxi-predict cost estimator on the task description provided.

Capture the current working directory, then locate the installed predictor and run it:

```bash
REPO_PATH="$(pwd)" && python -m predictor.predict "$ARGUMENTS" --models single --repo-path "$REPO_PATH"
```

If that fails because synaxi-predict isn't on the system Python path, try:

```bash
REPO_PATH="$(pwd)" && PREDICT_DIR="$(python -c "import predictor, os; print(os.path.dirname(os.path.dirname(predictor.__file__)))" 2>/dev/null || find ~ -name "predict.py" -path "*/predictor/*" -not -path "*/.venv/*" | head -1 | xargs dirname | xargs dirname)" && cd "$PREDICT_DIR" && source .venv/bin/activate 2>/dev/null; python -m predictor.predict "$ARGUMENTS" --models single --repo-path "$REPO_PATH"
```

After showing the output, give a one-sentence interpretation of the recommendation — specifically whether the cost difference justifies using the more expensive model, given the pass rate gap shown.
