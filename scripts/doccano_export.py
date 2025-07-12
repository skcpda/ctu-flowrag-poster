#!/usr/bin/env python
"""Export annotations from Doccano and convert to CTU-FlowRAG format."""

import json
import requests
from pathlib import Path
import argparse
from urllib.parse import urljoin


def export_project(project_id: int, doccano_url: str, username: str, password: str) -> list:
    """Download all annotations for a Doccano project."""
    session = requests.Session()
    # Login
    login_resp = session.post(urljoin(doccano_url, "/v1/auth-token/"), json={
        "username": username,
        "password": password,
    }, timeout=30)
    login_resp.raise_for_status()

    token = login_resp.json().get("token")
    session.headers.update({"Authorization": f"Token {token}"})

    export_resp = session.get(urljoin(doccano_url, f"/v1/projects/{project_id}/download?format=json"), timeout=60)
    export_resp.raise_for_status()
    return export_resp.json()


def convert_annotations(data: list) -> list:
    """Convert Doccano JSONL into CTU evaluation format."""
    converted = []
    for record in data:
        sentences = [s[1] for s in sorted(record["text"])] if isinstance(record["text"], list) else []
        boundaries = [int(idx) for idx, _ in record.get("labels", [])]
        converted.append({
            "doc_id": record.get("id"),
            "sentences": sentences,
            "boundaries": sorted(boundaries),
        })
    return converted


def main():
    parser = argparse.ArgumentParser(description="Export CTU annotations from Doccano")
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--doccano-url", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    raw = export_project(args.project_id, args.doccano_url, args.username, args.password)
    converted = convert_annotations(raw)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(converted, f, indent=2)

    print(f"✅ Exported {len(converted)} docs → {args.output}")


if __name__ == "__main__":
    main() 