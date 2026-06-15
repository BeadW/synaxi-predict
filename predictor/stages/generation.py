"""Generation stage – runs the core model inference or code generation."""

from __future__ import annotations

from typing import Any, Dict, Optional


class GenerationStage:
    """
    Simulates the model inference step. In a real system this would load a
    model (e.g., XGBoost, scikit‑learn, or a deep learning model) and run a
    prediction based on the gathered context.
    """

    def execute(
        self,
        input_data: Dict[str, Any],
        previous_output: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Return a mock prediction result.

        The previous_output is expected to contain a ``context`` key from the
        ``ResearchStage``.  Here we simply fabricate a numeric score.
        """
        context = previous_output.get("context", {}) if previous_output else {}
        # Dummy logic: combine the number of sources with a constant factor
        sources = len(context.get("sources", []))
        prediction_score = 0.6 + (sources * 0.1)
        return {"prediction": {"score": round(prediction_score, 3), "details": context}}
