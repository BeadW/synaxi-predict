"""Planning stage – determines the plan/strategy based on validated data."""

from __future__ import annotations

from typing import Any, Dict, Optional


class PlanningStage:
    """
    Produces a plan object that downstream stages can use.
    """

    def execute(
        self,
        input_data: Dict[str, Any],
        previous_output: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate a simple plan structure."""
        validated = previous_output.get("validated_data") if previous_output else None
        return {
            "plan": {
                "strategy": "basic",
                "steps": ["research", "generate", "verify"],
                "context": validated,
            }
        }