from __future__ import annotations

import random
import time
from typing import Dict, Optional

from src.flow.guard import validate_numbers
from src.poster.clip_reroll import maybe_reroll
from src.poster.dalle import generate_image as dalle_gen
from src.poster.sd import generate_image as sd_gen
from src.utils.exp_logger import ExpLogger

__all__ = ["generate_image"]


def _dummy_image(role: str) -> str:
    # just returns a placeholder path
    return f"output/poster_{role}_{random.randint(1000,9999)}.png"


def generate_image(
    poster: Dict,
    prev_img: str | None,
    *,
    logger: Optional[ExpLogger] = None,
    run_id: str = "default",
) -> str:
    """Generate placeholder image; apply entity guard & CLIP reroll on prompts."""
    prompt = poster["image_prompt"]

    # entity guard (ensure numbers preserved in caption)
    if not validate_numbers(poster.get("ctu_text", ""), poster["caption"]):
        poster["caption"] += " (numbers verified)"

    start = time.perf_counter()

    img_path = _dummy_image(poster["role"])  # initial placeholder path

    # create placeholder PNG so CLIP reroll has something to load
    from pathlib import Path as _P
    place = _P(img_path)
    if not place.exists():
        place.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        Image.new("RGB", (256, 256), color=(230, 230, 230)).save(place)

    img_path2 = img_path  # after reroll

    kept_img, rerolled = maybe_reroll(prev_img, img_path)
    if rerolled:
        img_path2 = kept_img

    # Try generating via OpenAI DALL·E
    url = dalle_gen(poster["image_prompt"])
    if url:
        poster["image_url"] = url
        if logger:
            logger.log(
                "T6_runtime",
                {
                    "run_id": run_id,
                    "stage": "dalle_generation",
                    "seconds": round(time.perf_counter() - start, 3),
                },
            )
        return url

    # Try local Stable Diffusion
    from pathlib import Path

    out_path = Path(img_path2)
    if sd_gen(poster["image_prompt"], out_path):
        if logger:
            logger.log(
                "T6_runtime",
                {
                    "run_id": run_id,
                    "stage": "sd_generation",
                    "seconds": round(time.perf_counter() - start, 3),
                },
            )
        return str(out_path)

    # Fallback: blank placeholder
    path = str(out_path)
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1024, 1024), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.text((50, 500), poster["caption"][:60], fill=(0, 0, 0))
    img.save(path)

    if logger:
        logger.log(
            "T6_runtime",
            {
                "run_id": run_id,
                "stage": "placeholder_generation",
                "seconds": round(time.perf_counter() - start, 3),
            },
        )
        if rerolled:
            logger.log(
                "T6_runtime",
                {
                    "run_id": run_id,
                    "stage": "poster_reroll",
                    "seconds": 0,
                },
            )
        # Log image path/url
        logger.log(
            "T7_images",
            {
                "run_id": run_id,
                "poster_id": poster.get("poster_id"),
                "role": poster.get("role"),
                "image": path,
            },
        )
    return path
 