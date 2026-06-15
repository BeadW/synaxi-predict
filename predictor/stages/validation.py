"""Validation stage – ensures input data conforms to expected schema."""

from __future__ import annotations

from typing import Any, Dict, Optional


class ValidationStage:
    """
    Performs minimal validation on incoming input data.

    In a real scenario this would validate against a JSON Schema,
    type hints, or domain-specific constraints.
    """

    def execute(
        self,
        input_data: Dict[str, Any],
        previous_output: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Validates and returns sanitized data for next stage."""
        payload = input_data.get("payload")
        if payload is None:
            raise ValueError("Input 'payload' is required")
        return {"validated_data": payload}