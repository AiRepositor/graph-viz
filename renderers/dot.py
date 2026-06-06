#!/usr/bin/env python3
"""
Graphviz DOT renderer — takes an abstract GraphModel and produces SVG output.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph_model import GraphModel, Node

logger = logging.getLogger(__name__)


# ── Node shape/style mapping ──────────────────────────────────────────────

NODE_STYLE = {
    "repo":         {"shape": "box3d",      "fill": "#16213e", "border": "#0f3460"},
    "skill":        {"shape": "folder",     "fill": "#1a3a2e", "border": "#2d6a4f"},
    "skill_group":  {"shape": "folder",     "fill": "#1a3a2e", "border": "#2d6a4f", "fontsize": "11"},
    "profile":      {"shape": "cylinder",   "fill": "#3a1a1a", "border": "#6a2d2d"},
    "config":       {"shape": "note",       "fill": "#2a2a1a", "border": "#6a6a2d"},
    "cron":         {"shape": "component",  "fill": "#1a2a3a", "border": "#2d4a6a"},
    "plugin":       {"shape": "box",        "fill": "#2a1a2a", "border": "#6a2d6a"},
    "tool":         {"shape": "box",        "fill": "#1a3a3a", "border": "#2d6a6a"},
    "session":      {"shape": "ellipse",    "fill": "#2a2a3a", "border": "#4a4a6a"},
    "unknown":      {"shape": "ellipse",    "fill": "#2a2a2a", "border": "#555555"},
}


def _node_style(node: Node) -> dict:
    """Pick style for a node, considering aggregated subtypes."""
    if node.type == "skill" and node.subtype == "grouped":
        return NODE_STYLE["skill_group"]
    return NODE_STYLE.get(node.type, NODE_STYLE["unknown"])


def _node_label(node: Node) -> str:
    """Build rich HTML-style label for a node."""
    parts = [f"<b>{node.name}</b>"]
    meta = node.metadata

    if node.type == "repo":
        branch = meta.get("branch", "")
        if branch:
            parts.append(f'<br/><font point-size="8" color="#666">⎇ {branch}</font>')
        commit_msg = meta.get("commit_msg", "")
        if commit_msg:
            msg_short = commit_msg[:40] + ("…" if len(commit_msg) > 40 else "")
            parts.append(f'<br/><font point-size="7" color="#555">{msg_short}</font>')
        if meta.get("has_uncommitted"):
            parts.append('<br/><font point-size="8" color="#ffaa00">⚠ uncommitted</font>')
        langs = meta.get("languages", [])
        if langs:
            parts.append(f'<br/><font point-size="7" color="#666">{" ".join(langs[:3])}</font>')

    elif node.type == "skill" and node.subtype == "grouped":
        count = meta.get("count", 0)
        parts.append(f'<br/><font point-size="9" color="#4caf50">● {count} skills</font>')

    elif node.type == "skill":
        cat = meta.get("category", "")
        if cat:
            parts.append(f'<br/><font point-size="8" color="#666">[{cat}]</font>')

    elif node.type == "config":
        skills_total = meta.get("skills_total", 0)
        profiles_c = meta.get("profiles_count", 0)
        cron_c = meta.get("cron_count", 0)
        sessions_c = meta.get("sessions_count", 0)
        info = []
        if skills_total:
            info.append(f"{skills_total} skills")
        if profiles_c:
            info.append(f"{profiles_c} profiles")
        if cron_c:
            info.append(f"{cron_c} cron")
        if sessions_c:
            info.append(f"{sessions_c} sessions")
        if info:
            parts.append(f'<br/><font point-size="7" color="#666">{" · ".join(info)}</font>')

    elif node.type == "cron":
        count = meta.get("count", 0)
        parts.append(f'<br/><font point-size="8" color="#666">{count} scheduled tasks</font>')

    return "".join(parts)


def render(model: GraphModel, output_path: str | None = None,
           engine: str | None = None) -> dict[str, Any]:
    """
    Render the graph model to SVG using Graphviz.

    Args:
        model: The graph model to render
        output_path: Path for output SVG (default: ~/graphs/<name>.svg)
        engine: Graphviz engine (auto-picked from model if not set)

    Returns:
        Dict with result info including svg_path, dot_path, node/edge stats.
    """
    try:
        import graphviz
    except ImportError:
        return {"success": False, "error": "graphviz Python package not installed"}

    if engine is None:
        engine = model.suggested_engine

    # Layout: dot for small graphs (<20), neato for medium, fdp for large
    n = len(model.nodes)
    auto_engine = "dot" if n <= 15 else ("neato" if n <= 40 else "fdp")
    if engine is None:
        engine = auto_engine

    dot = graphviz.Digraph(
        name="agent-graph",
        comment=f"Project Graph: {model.title}",
        format="svg",
        engine=engine,
        graph_attr={
            "rankdir": "TB",
            "bgcolor": "#0d1117",
            "fontcolor": "#c9d1d9",
            "fontname": "monospace",
            "label": f"  {model.title}  ",
            "labelloc": "t",
            "fontsize": "24",
            "dpi": "150",
            "splines": "true",
            "overlap": "false",
            "pad": "0.5",
            "nodesep": "0.6",
            "ranksep": "0.8",
            "style": "rounded",
            "penwidth": "0",
        },
        node_attr={
            "fontname": "monospace",
            "fontsize": "10",
            "penwidth": "1.5",
            "margin": "0.15,0.08",
        },
        edge_attr={
            "fontname": "monospace",
            "fontsize": "8",
            "fontcolor": "#8b949e",
        },
    )

    # ── Subgraphs (clusters) ──
    for cname, cluster in model.clusters.items():
        with dot.subgraph(name=f"cluster_{cluster.id}") as sub:
            sub.attr(
                label=f"  {cluster.name}  ",
                labeljust="l",
                fontcolor="#c9d1d9",
                fontsize="14",
                fontname="monospace-bold",
                style="filled",
                fillcolor=cluster.color,
                color="#30363d",
                penwidth="1",
                margin="16",
            )

            for node_id in sorted(cluster.node_ids):
                node = model.nodes.get(node_id)
                if node is None:
                    continue

                st = _node_style(node)
                label = f"<{_node_label(node)}>"

                sub.node(
                    node.id,
                    label=label,
                    shape=st["shape"],
                    style="filled",
                    fillcolor=st["fill"],
                    color=st["border"],
                    fontcolor="#c9d1d9",
                    fontsize=st.get("fontsize", "10"),
                    tooltip=_tooltip(node),
                )

    # ── Unclustered nodes ──
    clustered_ids = set()
    for cluster in model.clusters.values():
        clustered_ids.update(cluster.node_ids)

    for node in model.nodes.values():
        if node.id not in clustered_ids:
            st = _node_style(node)
            label = f"<{_node_label(node)}>"
            dot.node(
                node.id,
                label=label,
                shape=st["shape"],
                style="filled",
                fillcolor=st["fill"],
                color=st["border"],
                fontcolor="#c9d1d9",
                fontsize=st.get("fontsize", "10"),
                tooltip=_tooltip(node),
            )

    # ── Edges ──
    for edge in model.edges:
        st = edge.get_style()
        label = f"  {edge.label or st['label']}  " if (edge.label or st['label']) else None
        dot.edge(
            edge.source, edge.target,
            label=label,
            color=st["color"],
            style=st["style"],
            fontsize="7",
            fontcolor=st["color"],
            penwidth=str(max(0.5, edge.weight * 0.6)),
        )

    # ── Output path ──
    if output_path is None:
        graphs_dir = Path.home() / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = str(graphs_dir / f"project-graph-{ts}.svg")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    render_path = str(out.with_suffix(""))

    try:
        rendered = dot.render(filename=render_path, cleanup=True)
    except Exception as e:
        return {"success": False, "error": f"Graphviz render failed: {e}"}

    dot_path = out.with_suffix(".gv")
    dot.save(filename=str(dot_path))

    stats = model.stats
    return {
        "success": True,
        "svg_path": rendered,
        "dot_path": str(dot_path),
        "engine": engine,
        "stats": stats,
    }


def _tooltip(node: Node) -> str:
    """Build a tooltip string for a node."""
    meta = node.metadata
    parts = [f"Type: {node.type}"]
    if node.subtype:
        parts.append(node.subtype)
    if meta.get("path"):
        parts.append(f"Path: {meta['path']}")
    if meta.get("remote"):
        parts.append(f"Remote: {meta['remote']}")
    if meta.get("count"):
        parts.append(f"Count: {meta['count']}")
    return "  |  ".join(parts)
