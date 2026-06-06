#!/usr/bin/env python3
"""
GraphViz — see what you're working on.

A single command to map your project landscape: git repos, Hermes skills,
config, and anything else — all in one smart SVG graph.

Usage:
  python3 app.py                         → repos only, terminal summary
  python3 app.py --deep                  → repos + agent artifacts
  python3 app.py --name my-projects      → custom filename
  python3 app.py --engine neato          → different layout
  python3 app.py --dirs ~/projects,~/work → scan custom dirs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from graph_model import GraphModel
from collectors.git import collect as collect_git
from collectors.hermes import collect as collect_hermes
from intelligence.classify import classify
from renderers.dot import render as render_dot

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(message)s",
)
logger = logging.getLogger("graph-viz")


# ── Terminal output helpers ──────────────────────────────────────────────

def _print_summary(model: GraphModel, elapsed: float) -> str:
    """Build a readable terminal summary of the graph."""
    lines = []
    title = "  📊  Project Landscape"
    lines.append("")
    lines.append(f"  ┌{'─' * 55}┐")
    lines.append(f"  │ {title:<53} │")
    lines.append(f"  └{'─' * 55}┘")

    # Group nodes by type
    repos = [n for n in model.nodes.values() if n.type == "repo"]
    skills = [n for n in model.nodes.values() if n.type == "skill"]
    profiles = [n for n in model.nodes.values() if n.type == "profile"]
    others = [n for n in model.nodes.values()
              if n.type not in ("repo", "skill", "profile")]

    if repos:
        lines.append("")
        lines.append("  \033[1mRepos\033[0m")
        for r in repos:
            meta = r.metadata
            langs = ", ".join(meta.get("languages", []))[:30]
            branch = meta.get("branch", "")
            commit = meta.get("commit_msg", "")[:35]
            cluster = r.cluster or ""

            # Color by cluster
            tag = f" \033[90m[{cluster}]\033[0m" if cluster else ""
            lang_tag = f" \033[90m({langs})\033[0m" if langs else ""
            branch_tag = f" \033[90m⎇ {branch}\033[0m" if branch else ""
            dirty = " \033[33m⚠\033[0m" if meta.get("has_uncommitted") else ""

            lines.append(f"    • \033[1m{r.name}\033[0m{tag}{lang_tag}{branch_tag}{dirty}")

            # Show commit on second line if available
            if commit:
                lines.append(f"      \033[90m{commit}\033[0m")

    if skills:
        lines.append("")
        lines.append(f"  \033[1mAgent Skills\033[0m  \033[90m({len(skills)} groups)\033[0m")
        for s in skills:
            meta = s.metadata
            count = meta.get("count", "")
            count_tag = f" \033[90m×{count}\033[0m" if count else ""
            lines.append(f"    • {s.name}{count_tag}")

    if profiles:
        lines.append("")
        lines.append("  \033[1mProfiles\033[0m")
        for p in profiles:
            lines.append(f"    • {p.name}")

    if others:
        lines.append("")
        lines.append(f"  \033[1mOther\033[0m  \033[90m({len(others)} nodes)\033[0m")
        for o in others:
            lines.append(f"    • {o.name}")

    # Stats bar
    stats = model.stats
    clusters_info = f", {stats['total_clusters']} groups" if stats['total_clusters'] else ""
    lines.append("")
    lines.append(f"  \033[90m{stats['total_nodes']} nodes · {stats['total_edges']} edges{clusters_info} · {elapsed:.1f}s\033[0m")
    lines.append("")

    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────

def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Map your project landscape — repos, agent artifacts, and more.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--name", "-n", type=str, default=None,
                    help="Name for the graph (used in filename)")
    p.add_argument("--no-agent", action="store_true",
                    help="Skip Hermes agent artifacts (skills, config, profiles)")
    p.add_argument("--dirs", type=str, default=None,
                    help="Scan these dirs for repos (default: ~)")
    p.add_argument("--engine", "-e", type=str, default=None,
                    choices=["dot", "neato", "fdp"],
                    help="Layout: dot (hierarchy), neato (spring), fdp (large graphs)")
    p.add_argument("--no-llm", action="store_true",
                    help="Skip LLM classification, use heuristics")
    p.add_argument("--json", action="store_true",
                    help="Output JSON model (no SVG)")
    p.add_argument("--ui", action="store_true",
                    help="Write JSON for the interactive UI and print the UI path")
    p.add_argument("--max-depth", type=int, default=5,
                    help="Max depth for repo search (default: 5)")
    return p


def run(args: argparse.Namespace) -> dict:
    """Run the graph pipeline."""
    scan_dirs = [d.strip() for d in args.dirs.split(",")] if args.dirs else None
    t0 = datetime.now(timezone.utc)

    # ── Collect repos ──
    print("  \033[90mScanning repos...\033[0m", end=" ", flush=True)
    model = collect_git(root_dirs=scan_dirs, max_depth=args.max_depth)
    total_repos = len([n for n in model.nodes.values() if n.type == 'repo'])
    print(f"\033[92m✓\033[0m ({total_repos} found)")

    # ── Collect agent artifacts (default: yes, unless --no-agent) ──
    if not args.no_agent:
        print("  \033[90mScanning agent state...\033[0m", end=" ", flush=True)
        collect_hermes(model=model, aggregated=True)
        total_skills = sum(
            n.metadata.get("count", 0) or 1
            for n in model.nodes.values() if n.type == "skill"
        )
        total_profiles = len([n for n in model.nodes.values() if n.type == "profile"])
        print(f"\033[92m✓\033[0m ({total_skills} skills, {total_profiles} profiles)")

    if not model.nodes:
        print("\n  \033[33mNo repos found to graph.\033[0m")
        return {"success": False, "error": "No nodes found"}

    # ── Classify ──
    print(f"  \033[90mClassifying{' with AI' if not args.no_llm else ''}...\033[0m", end=" ", flush=True)
    model = classify(model)
    print(f"\033[92m✓\033[0m ({model.stats['total_clusters']} groups)")

    # ── Output ──
    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()

    if args.json:
        print(_print_summary(model, elapsed))
        return {"success": True, "model": model.to_dict()}

    # Always write JSON data for the UI
    name = args.name or "project-graph"
    json_path = str(Path.home() / "graphs" / f"{name}.json")
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(model.to_dict(), f, indent=2)

    if args.ui:
        ui_path = str(Path.home() / "graph-viz-ui" / "out" / "index.html")
        ui_json_path = Path.home() / "graph-viz-ui" / "out" / "data.json"
        # Copy JSON into the UI build
        ui_json_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(json_path, str(ui_json_path))
        print(f"  \\033[90mJSON data written to {json_path}\\033[0m")
        print(f"  \\033[1mOpen UI →\\033[0m \\033[94mfile://{ui_path}\\033[0m")
        print(_print_summary(model, elapsed))
        return {"success": True, "json_path": json_path, "model": model.to_dict()}

    # Determine output path
    output_path = str(Path.home() / "graphs" / f"{name}.svg")

    print(f"  \033[90mRendering ({args.engine or model.suggested_engine})...\033[0m", end=" ", flush=True)
    result = render_dot(model, output_path=output_path, engine=args.engine)
    render_elapsed = (datetime.now(timezone.utc) - t0).total_seconds()

    if result.get("success"):
        svg = Path(result["svg_path"])
        print(f"\033[92m✓\033[0m")
        print("")
        print(f"  \033[1mGraph ready →\033[0m \033[94mfile://{svg.resolve()}\033[0m")
        print(_print_summary(model, render_elapsed))

    return result


def main() -> int:
    parser = build_cli()
    args = parser.parse_args()

    try:
        result = run(args)
    except KeyboardInterrupt:
        print("\n  Interrupted.")
        return 130
    except Exception as e:
        logger.exception("Error")
        print(f"\n  \033[31mError: {e}\033[0m")
        return 1

    if args.json and result.get("success"):
        print(json.dumps(result.get("model", {}), indent=2))

    if not result.get("success"):
        print(f"\n  \033[31mError: {result.get('error', 'Unknown')}\033[0m")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
