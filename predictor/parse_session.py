"""
Parse a Claude Code subagent session JSONL and return exact metrics as JSON.

Usage:
    python -m predictor.parse_session <agent-id> [repo-cwd]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from predictor.predict import _DEFAULT_PRICING, _PRICING


def find_session_file(agent_id: str, cwd: Path) -> Path | None:
    # Claude Code encodes paths by replacing every "/" with "-" (leading slash becomes leading dash)
    encoded = str(cwd).replace("/", "-")
    projects_dir = Path.home() / ".claude" / "projects" / encoded
    matches = list(projects_dir.rglob(f"agent-{agent_id}.jsonl"))
    return matches[0] if matches else None


def parse_session(path: Path) -> dict:
    turns = 0
    total_input = 0
    total_output = 0
    total_cache_read = 0
    total_cache_creation = 0
    prompt_tokens_last = 0
    model = None

    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("type") != "assistant":
            continue

        msg = d.get("message", {})
        usage = msg.get("usage", {})
        content = msg.get("content", [])

        if model is None:
            model = msg.get("model")

        inp = usage.get("input_tokens", 0)
        out = usage.get("output_tokens", 0)
        cr  = usage.get("cache_read_input_tokens", 0)
        cc  = usage.get("cache_creation_input_tokens", 0)

        total_input        += inp
        total_output       += out
        total_cache_read   += cr
        total_cache_creation += cc

        tool_uses = [c for c in content if isinstance(c, dict) and c.get("type") == "tool_use"]
        if tool_uses:
            turns += 1

        # Context window size at this call (for prompt_tokens_raw convention)
        if inp + cr + cc > 0:
            prompt_tokens_last = inp + cr + cc

    p_in, p_out, p_cw, p_cr = _PRICING.get(model, _DEFAULT_PRICING) if model else _DEFAULT_PRICING
    cost = (
        total_input        * p_in +
        total_cache_creation * p_cw +
        total_cache_read   * p_cr +
        total_output       * p_out
    )

    return {
        "turns":              turns,
        "completion_tokens":  total_output,
        "prompt_tokens_last": prompt_tokens_last,
        "cache_read_tokens":  total_cache_read,
        "cache_creation_tokens": total_cache_creation,
        "model":              model,
        "cost_usd":           round(cost, 6),
        "session_file":       str(path),
    }


def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: parse-session <agent-id> [repo-cwd]"}),
              file=sys.stderr)
        sys.exit(1)

    agent_id = sys.argv[1]
    cwd = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.cwd()

    session_file = find_session_file(agent_id, cwd)
    if not session_file:
        print(json.dumps({"error": f"No session file for agent {agent_id} under {cwd}"}),
              file=sys.stderr)
        sys.exit(1)

    print(json.dumps(parse_session(session_file)))


if __name__ == "__main__":
    main()
