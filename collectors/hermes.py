#!/usr/bin/env python3
"""
Hermes agent state collector — discovers skills, profiles, config, cron, plugins.

Supports two modes:
  - full: individual nodes for every artifact (100+ skill nodes)
  - aggregated (default): skills grouped by category, config as summary
"""

from __future__ import annotations

import logging
import os
import sqlite3
from collections import Counter
from pathlib import Path
from graph_model import Node, GraphModel, _safe_id

logger = logging.getLogger(__name__)


def _get_hermes_home() -> Path:
    env = os.environ.get("HERMES_HOME")
    if env:
        return Path(env)
    return Path.home() / ".hermes"


def collect(hermes_home: str | None = None,
            model: GraphModel | None = None,
            aggregated: bool = True) -> GraphModel:
    """
    Scan Hermes agent state and add artifacts to the model.

    Args:
        hermes_home: Path to Hermes config directory (default: ~/.hermes)
        model: Existing GraphModel to populate (creates new one if None)
        aggregated: If True, group skills by category into single nodes
                    with count badges. If False, add every skill individually.

    Returns:
        GraphModel with agent artifact nodes added.
    """
    home = Path(hermes_home) if hermes_home else _get_hermes_home()

    if model is None:
        model = GraphModel()

    if not home.is_dir():
        logger.warning("Hermes home not found: %s", home)
        return model

    # ── Config ──
    config_path = home / "config.yaml"
    config_size = config_path.stat().st_size if config_path.is_file() else 0
    config_node = Node.config(f"⚙ config  ({config_size // 1024}k)")
    config_node.metadata["path"] = str(config_path)
    model.add_node(config_node)

    # ── Skills ──
    skills_dir = home / "skills"
    skill_count = 0
    if skills_dir.is_dir():
        if aggregated:
            # Group skills by category
            categories: Counter = Counter()
            for skill_file in skills_dir.rglob("SKILL.md"):
                cat = skill_file.parent.parent.name
                if cat == "skills":  # top-level
                    cat = "_uncategorized"
                categories[cat] += 1
                skill_count += 1

            for cat_name, count in sorted(categories.items()):
                display = cat_name.replace("-", " ").replace("_", " ").title()
                node_id = _safe_id(f"skills:{cat_name}")
                model.add_node(Node(
                    id=node_id,
                    name=f"{display} ({count})",
                    type="skill",
                    subtype="grouped",
                    metadata={"category": cat_name, "count": count},
                ))
                model.add_edge("config", node_id, kind="contains",
                               label=f"{count} skills")

            # Add a total skills count to config metadata
            config_node.metadata["skills_total"] = skill_count
        else:
            # Individual skills (original behavior)
            for skill_file in sorted(skills_dir.rglob("SKILL.md")):
                cat = skill_file.parent.parent.name
                if cat == "skills":
                    cat = ""
                name = skill_file.parent.name
                model.add_node(Node.skill(name=name, category=cat))
                model.add_edge("config", _safe_id(f"skill:{name}"),
                               kind="contains", label="loads")
                skill_count += 1

    # ── Profiles ──
    profiles_dir = home / "profiles"
    profile_count = 0
    if profiles_dir.is_dir():
        for profile_dir in sorted(profiles_dir.iterdir()):
            if profile_dir.is_dir():
                model.add_node(Node.profile(profile_dir.name))
                model.add_edge(
                    _safe_id(f"profile:{profile_dir.name}"), "config",
                    kind="uses", label="configured by",
                )
                profile_count += 1

    if profile_count > 0 or aggregated:
        config_node.metadata["profiles_count"] = profile_count

    # ── Cron jobs ──
    state_db = home / "state.db"
    cron_count = 0
    if state_db.is_file():
        try:
            conn = sqlite3.connect(str(state_db))
            row = conn.execute("SELECT COUNT(*) FROM cron_jobs").fetchone()
            cron_count = row[0] if row else 0
            conn.close()
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            pass

    if cron_count > 0:
        model.add_node(Node.cron(cron_count))
        model.add_edge("cron", "config", kind="uses",
                       label=f"{cron_count} scheduled")
    config_node.metadata["cron_count"] = cron_count

    # ── Plugins ──
    plugins_dir = home / "plugins"
    plugin_count = 0
    if plugins_dir.is_dir() and not aggregated:
        for d in sorted(plugins_dir.iterdir()):
            if d.is_dir() and not d.name.startswith("__"):
                model.add_node(Node.plugin(d.name))
                model.add_edge(
                    _safe_id(f"plugin:{d.name}"), "config",
                    kind="uses", label="plugin",
                )
                plugin_count += 1
    config_node.metadata["plugins_count"] = (plugin_count
        or len([d for d in plugins_dir.iterdir() if d.is_dir() and not d.name.startswith("__")])
        if plugins_dir.is_dir() else 0)

    # ── Tools (from hermes-agent source) ──
    agent_tools_dir = home / "hermes-agent" / "tools"
    tool_count = 0
    if agent_tools_dir.is_dir() and not aggregated:
        for f in sorted(agent_tools_dir.glob("*_tool.py")):
            tool_name = f.stem
            model.add_node(Node.tool(tool_name))
            model.add_edge("config", _safe_id(f"tool:{tool_name}"),
                           kind="contains")
            tool_count += 1
    config_node.metadata["tools_count"] = (
        tool_count or len(list(agent_tools_dir.glob("*_tool.py")))
        if agent_tools_dir.is_dir() else 0
    )

    # ── Sessions ──
    session_count = 0
    if state_db.is_file():
        try:
            conn = sqlite3.connect(str(state_db))
            row = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
            session_count = row[0] if row else 0
            conn.close()
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            pass

    config_node.metadata["sessions_count"] = session_count
    if session_count > 0 and not aggregated:
        model.add_node(Node(
            id="sessions",
            name=f"{session_count} sessions",
            type="session",
            metadata={"count": session_count},
        ))
        model.add_edge("sessions", "config", kind="owned_by",
                       label="managed by")

    return model
