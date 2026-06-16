"""
Basic Prediction Pipeline Implementation

This module defines a simple, extensible pipeline that can be used as the
foundation for the overall prediction system.  It wires together the typical
stages:

1.  Input validation / data preparation
2.  Planning / requirement gathering
3.  Research & context gathering
4.  Code generation (or model inference)
5.  Verification / testing
6.  Synthesis & reporting

The pipeline is deliberately lightweight – it only provides the scaffolding
and orchestration logic.  Concrete implementations of each stage should be
provided by separate modules (e.g. `planner.py`, `researcher.py`, …) that
this pipeline can import and invoke.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Prevent duplicate handlers when this module is reloaded
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------
class PredictionPipeline:
    """
    High‑level pipeline that runs through the predefined stages.

    The pipeline is deliberately generic – each stage is represented by a
    pluggable component that can be swapped out or extended.
    """

    def __init__(self, stages: Optional[List[Any]] = None):
        """
        Parameters
        ----------
        stages:
            List of stage objects that implement the ``execute`` method.
            If omitted, a default ordering is used (see ``default_stages``).
        """
        self.stages = stages if stages is not None else self.default_stages()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the pipeline on ``input_data`` and return the final result.

        The ``input_data`` dictionary should contain at least a ``payload`` key
        with the raw data to be processed, but any additional keys can be used
            by downstream stages.

        Returns
        -------
        dict
            The result dictionary produced by the final stage.
        """
        logger.info("🚀 Starting prediction pipeline")
        result: Dict[str, Any] = {"stage_status": [], "output": None}

        for stage in self.stages:
            stage_name = stage.__class__.__name__
            logger.info(f"▶️ Executing stage: {stage_name}")

            try:
                stage_output = stage.execute(input_data, result.get("output"))
                result["output"] = stage_output
                result["stage_status"].append((stage_name, "success"))
                logger.info(f"✅ Stage {stage_name} completed successfully")
            except Exception as exc:
                logger.exception(f"❌ Stage {stage_name} failed: {exc}")
                result["stage_status"].append((stage_name, f"failed: {exc}"))
                # Continue with next stage – you can change this behaviour.
                continue

        logger.info("🏁 Prediction pipeline finished")
        return result

    # -----------------------------------------------------------------------
    # Default stage ordering – can be overridden by passing a custom list
    # -----------------------------------------------------------------------
    @staticmethod
    def default_stages() -> List[Any]:
        """
        Returns the default list of stage objects.  Sub‑classes or users can
        replace this with a custom ordering or inject mock stages for testing.
        """
        from .stages import (
            ValidationStage,
            PlanningStage,
            ResearchStage,
            GenerationStage,
            VerificationStage,
        )

        return [
            ValidationStage(),
            PlanningStage(),
            ResearchStage(),
            GenerationStage(),
            VerificationStage(),
        ]


# ---------------------------------------------------------------------------
# Convenience entry‑point
# ---------------------------------------------------------------------------
def main() -> None:
    """
    Simple command‑line entry point for quick manual testing.

    Example
    -------
    $ python -m predictor.pipeline --data '{"payload": "sample"}'
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run the basic prediction pipeline")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="JSON string containing the input payload",
    )
    args = parser.parse_args()

    try:
        payload = json.loads(args.data)
    except json.JSONDecodeError as exc:
        logger.error(f"Invalid JSON supplied: {exc}")
        sys.exit(1)

    pipeline = PredictionPipeline()
    outcome = pipeline.run(payload)
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()