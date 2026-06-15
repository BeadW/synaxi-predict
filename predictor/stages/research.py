"""Research stage – gathers external context or performs lookups."""

from __future__ import annotations

from typing import Any, Dict, Optional


class ResearchStage:
    """
    Simulates a research phase.  In a real system this would query
    external services, load datasets, or run literature searches.
    """

    def execute(
        self,
        input_data: Dict[str, Any],
        previous_output: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Return a mock context object."""
        plan = previous_output.get("plan", {}) if previous_output else {}
        return {
            "context": {
                "sources": ["mock_source_1", "mock_source_2"],
                "strategy": plan.get("strategy", "unknown"),
            }
        }