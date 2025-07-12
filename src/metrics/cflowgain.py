"""C-FlowGain: composite poster-quality metric.

We combine three components:
1. **CLIP similarity** between prompt and generated image (0-1)
2. **Flow order penalty** – posters should follow canonical role order.
3. **Safety penalty** – NSFW likelihood (from OpenAI or a local classifier).

For now we expose a single `compute_cflowgain` function that takes the
pipeline `results` dict and returns an overall float score.
"""

from __future__ import annotations

from typing import List, Dict


def compute_cflowgain(pipeline_results: Dict, role_order: List[str] | None = None) -> float:
    """Compute C-FlowGain for a pipeline run.

    Args:
        pipeline_results: Dict produced by `CTUFlowRAGPipeline.save_results()`
            (must contain `posters` list with `role` & optional `clip_score` / `nsfw_score`).
        role_order: Expected ordering of roles. Default takes synthesizer default.

    Returns:
        Scalar score ∈ [0,1] where higher is better.
    """
    posters = pipeline_results.get("posters", [])
    if not posters:
        return 0.0

    # 1) CLIP similarity average
    clip_scores = [p.get("clip_score", 0.0) for p in posters]
    clip_component = sum(clip_scores) / len(clip_scores) if clip_scores else 0.0

    # 2) Flow order – reward monotonic role indices
    if role_order is None:
        role_order = [
            "target_pop",
            "eligibility",
            "benefits",
            "procedure",
            "timeline",
            "contact",
            "misc",
        ]
    role_to_idx = {r: i for i, r in enumerate(role_order)}
    indices = [role_to_idx.get(p["role"], len(role_order)) for p in posters]
    order_violations = sum(1 for i in range(1, len(indices)) if indices[i] < indices[i - 1])
    flow_component = 1.0 - order_violations / max(1, len(indices) - 1)

    # 3) NSFW penalty (assume poster["nsfw_score"] ∈ [0,1] where 1 = definitely NSFW)
    nsfw_scores = [p.get("nsfw_score", 0.0) for p in posters]
    safety_penalty = sum(nsfw_scores) / len(nsfw_scores)

    # Combine (weights can be tuned)
    score = 0.5 * clip_component + 0.4 * flow_component - 0.3 * safety_penalty
    return max(0.0, min(1.0, score)) 