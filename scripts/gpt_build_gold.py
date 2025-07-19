from __future__ import annotations
"""Batch-label CTU roles with OpenAI GPT and write gold JSON.

Usage::
    export OPENAI_API_KEY=sk-...
    python scripts/gpt_build_gold.py \
        --input  output/pipeline_results_dedup.json \
        --output gold_final/seg_role/dev_roles.json \
        --num-schemes 150 \
        --model gpt-4o-mini

Output format is identical to pipeline results (top-level `ctus` list) so
`eval_seg_role.py` can consume it unmodified.
"""

import argparse, asyncio, json, os, random
from pathlib import Path
from typing import Dict, List

import aiohttp
from openai import AsyncOpenAI

SYSTEM_PROMPT = (
    "You are an annotator. Read each CTU and assign ONE role label from: "
    "benefits, eligibility, procedure, timeline, contact, target_pop, misc. "
    "Return JSON list: [{\"ctu_id\": int, \"role\": str}, …]."
)


def _sample_schemes(ctus: List[Dict], k: int, chunk_size: int = 400) -> Dict[str, List[Dict]]:
    """Return *k* scheme->CTU mapping.

    If CTUs lack a scheme/doc id we break them into pseudo‐schemes of
    at most *chunk_size* CTUs so that GPT calls stay small and logs remain
    readable.
    """
    random.shuffle(ctus)

    grouped: Dict[str, List[Dict]] = {}
    for c in ctus:
        gid = c.get("scheme_id") or c.get("doc_id")
        if not gid:
            # Create synthetic chunk id based on running count
            gid = f"chunk_{len(grouped)//10}"  # coarse bucket label
        grouped.setdefault(gid, []).append(c)

    # Further split any oversized group into fixed chunks
    expanded: Dict[str, List[Dict]] = {}
    for gid, lst in grouped.items():
        if len(lst) <= chunk_size:
            expanded[gid] = lst
        else:
            for i in range(0, len(lst), chunk_size):
                expanded[f"{gid}_{i//chunk_size}"] = lst[i : i + chunk_size]

    # Return first *k* groups
    keys = list(expanded.keys())[:k]
    return {k: expanded[k] for k in keys}


async def _label_scheme(client: AsyncOpenAI, model: str, scheme_ctus: List[Dict]) -> Dict[int, str]:
    chunks = [scheme_ctus[i : i + 40] for i in range(0, len(scheme_ctus), 40)]
    mapping: Dict[int, str] = {}
    async for chunk in _async_map(lambda ch: _call_chat(client, model, ch), chunks):
        mapping.update(chunk)
    return mapping


async def _call_chat(client: AsyncOpenAI, model: str, ctus: List[Dict]) -> Dict[int, str]:
    text = "\n".join(f"{c['ctu_id']}. {c['text']}" for c in ctus)
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0,
    )
    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
        return {int(r["ctu_id"]): r["role"] for r in data if "ctu_id" in r and "role" in r}
    except Exception:
        # fallback: mark everything misc
        return {c["ctu_id"]: "misc" for c in ctus}


async def _async_map(func, iterable):
    for item in iterable:
        yield await func(item)


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-schemes", type=int, default=150)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--chunk-size", type=int, default=400, help="Max CTUs per synthetic scheme/chunk")
    p.add_argument("--batch-size", type=int, default=40, help="CTUs per single GPT call (prompt)")
    p.add_argument("--log-dir", type=Path, default=Path("logs/gpt_gold"), help="Directory to save raw GPT outputs per scheme")
    args = p.parse_args()

    all_ctus = json.loads(args.input.read_text())["ctus"]
    sample = _sample_schemes(all_ctus, args.num_schemes, chunk_size=args.chunk_size)

    client = AsyncOpenAI()
    labelled: List[Dict] = []
    args.log_dir.mkdir(parents=True, exist_ok=True)
    for scheme, ctus in sample.items():
        log_fp = args.log_dir / f"{scheme}.json"

        if log_fp.exists():
            print(f"✅  {scheme} already done – skipping")
            try:
                existing = json.loads(log_fp.read_text())
                for cid, role in existing.items():
                    for c in ctus:
                        if c["ctu_id"] == int(cid):
                            cc = c.copy()
                            cc["role"] = role
                            labelled.append(cc)
            except Exception:
                pass  # corrupted log → fall through to re-annotate
            continue

        print(f"🔤  Annotating scheme {scheme} (CTUs {len(ctus)}) …")
        # override batch size
        global _BATCH_SIZE
        _BATCH_SIZE = args.batch_size
        mapping = await _label_scheme(client, args.model, ctus)
        log_fp.write_text(json.dumps(mapping, indent=2))

        for c in ctus:
            cc = c.copy()
            cc["role"] = mapping.get(c["ctu_id"], "misc")
            labelled.append(cc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"ctus": labelled}, indent=2))
    print("Saved", len(labelled), "labelled CTUs →", args.output)


if __name__ == "__main__":
    asyncio.run(main()) 