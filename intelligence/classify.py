#!/usr/bin/env python3
"""
LLM-assisted classification engine.

Classifies nodes (repos, skills, etc.) into meaningful categories using an
LLM. Falls back to heuristic classification when no API key is available.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

from graph_model import GraphModel, Node

logger = logging.getLogger(__name__)

# ── Classification taxonomy ──────────────────────────────────────────────

CATEGORIES = {
    "web_app":        "Web application (frontend, backend, or full-stack)",
    "cli_tool":       "Command-line interface tool or script",
    "ml_ai":          "Machine learning / AI model, training, or inference",
    "agent_tool":     "Agent framework tool, plugin, or skill",
    "library":        "Reusable library or package",
    "config":         "Configuration or dotfiles",
    "documentation":  "Documentation, wiki, or knowledge base",
    "infrastructure": "Infrastructure, DevOps, CI/CD, containers",
    "data_pipeline":  "Data processing, ETL, or analytics pipeline",
    "game":           "Game or game engine",
    "mobile_app":     "Mobile application (Android, iOS)",
    "api_service":    "API server or microservice",
    "learning":       "Learning materials, tutorials, experiments",
    "other":          "Other / unclassified",
}

# Sub-categories with more detail
SUB_CATEGORIES = {
    "nextjs":      "Next.js application",
    "react":       "React application",
    "vue":         "Vue.js application",
    "fastapi":     "FastAPI backend",
    "django":      "Django backend",
    "flask":       "Flask backend",
    "pytorch":     "PyTorch ML model",
    "rust_cli":    "Rust CLI tool",
    "go_service":  "Go microservice",
    "node_api":    "Node.js API",
    "python_lib":  "Python library",
    "agent_skill": "Hermes agent skill",
    "agent_tool":  "Hermes tool module",
    "agent_plugin":"Hermes plugin",
    "docs_site":   "Documentation site",
}


def _build_classification_prompt(nodes: list[Node]) -> str:
    """Build a structured prompt for the LLM to classify all nodes."""
    lines = [
        "You are a graph intelligence engine. Classify each item into one of the following categories.\n",
        "CATEGORIES (pick the single best match):",
    ]
    for key, desc in sorted(CATEGORIES.items()):
        lines.append(f"  - {key}: {desc}")

    lines.extend([
        "\nSUB-CATEGORIES (pick the single best match or leave empty if none fit):",
    ])
    for key, desc in sorted(SUB_CATEGORIES.items()):
        lines.append(f"  - {key}: {desc}")

    lines.extend([
        "\nFor each item, also suggest a CLUSTER — a higher-level group name (2-5 words).",
        "Nodes in the same cluster will be grouped together visually.",
        "Good clusters name real groupings: 'Web Frontends', 'ML Models', 'System Tools', 'Documentation', etc.\n",
        "\nRespond ONLY with a JSON array. No explanation, no markdown. Example:",
        """[
  {"id": "repo:my-app", "category": "web_app", "subcategory": "nextjs", "cluster": "Web Frontends"},
  {"id": "repo:ml-train", "category": "ml_ai", "subcategory": "pytorch", "cluster": "ML Models"}
]\n""",
        "\nITEMS TO CLASSIFY:",
    ])

    for node in nodes:
        meta = node.metadata
        langs = ", ".join(meta.get("languages", []))
        parts = [f'  id="{node.id}"']
        if langs:
            parts.append(f'languages=[{langs}]')
        if meta.get("remote"):
            parts.append(f'remote={meta["remote"]}')
        if meta.get("path"):
            # Only show the last 2 path parts
            path_parts = meta["path"].split("/")
            parts.append(f'path=.../{".../".join(path_parts[-2:])}')
        lines.append("  " + "  ".join(parts))

    return "\n".join(lines)


def _get_api_key() -> str:
    """Get an API key from env vars or ~/.hermes/.env file.

    Returns (api_key, base_url, model) tuple.
    """
    # Try DeepSeek first (user's primary provider), then fallback chain
    env_checks = [
        ("DEEPSEEK_API_KEY",        "https://api.deepseek.com",                "deepseek-chat"),
        ("OPENAI_API_KEY",          "https://api.openai.com/v1",              "gpt-4o-mini"),
        ("OPENROUTER_API_KEY",      "https://openrouter.ai/api/v1",           "openai/gpt-4o-mini"),
        ("ANTHROPIC_API_KEY",       "https://api.anthropic.com/v1",           "claude-3-haiku-20240307"),
    ]

    # First check env vars
    for env_var, base_url, model in env_checks:
        val = os.environ.get(env_var, "").strip()
        if val:
            return (val, base_url, model)

    # Fall back to reading ~/.hermes/.env
    env_path = Path.home() / ".hermes" / ".env"
    if env_path.is_file():
        try:
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("\"'")
                    for env_var, base_url, model in env_checks:
                        if key == env_var and value:
                            return (value, base_url, model)
        except OSError:
            pass

    return ("", "", "")


def _call_llm(prompt: str) -> str | None:
    """Call an OpenAI-compatible API for classification."""
    api_key, base_url, model = _get_api_key()

    if not api_key:
        logger.info("No API key found for LLM classification, using heuristic fallback")
        return None

    # Override from env if set
    classifier_base_url = os.environ.get("CLASSIFIER_BASE_URL", base_url)
    classifier_model = os.environ.get("CLASSIFIER_MODEL", model)

    url = f"{classifier_base_url.rstrip('/')}/chat/completions"
    payload = json.dumps({
        "model": classifier_model,
        "messages": [
            {
                "role": "system",
                "content": "You are a graph intelligence engine that classifies software artifacts. "
                           "Respond ONLY with valid JSON. No explanation, no markdown formatting.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
    }).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        content = result["choices"][0]["message"]["content"].strip()
        # Strip any markdown fences
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            content = content.rsplit("```", 1)[0]
        return content.strip()
    except (urllib.error.URLError, json.JSONDecodeError, KeyError,
            TimeoutError, OSError) as e:
        logger.warning("LLM classification API call failed: %s", e)
        return None


def _heuristic_classify(node: Node) -> tuple[str, str, str]:
    """Fallback heuristic classification when LLM is unavailable."""
    name = node.name.lower()
    meta = node.metadata
    languages = meta.get("languages", [])

    # Type-based
    if node.type == "config":
        return ("config", "", "Configuration")
    if node.type == "cron":
        return ("config", "", "Automation")
    if node.type == "session":
        return ("other", "", "Agent State")
    if node.type == "profile":
        return ("config", "", "Agent Profiles")
    if node.type == "plugin":
        return ("agent_tool", "agent_plugin", "Agent Plugins")
    if node.type == "tool":
        return ("agent_tool", "agent_tool", "Agent Tools")

    # Skill type
    if node.type == "skill":
        category = meta.get("category", "")
        if "ml" in name or "ai" in name:
            return ("ml_ai", "agent_skill", "ML/AI Skills")
        elif "devops" in name or "infra" in name:
            return ("infrastructure", "agent_skill", "DevOps Skills")
        elif "graph" in name or "viz" in name:
            return ("web_app", "agent_skill", "Visualization Tools")
        elif "data" in name or "research" in name:
            return ("data_pipeline", "agent_skill", "Data Skills")
        return ("agent_tool", "agent_skill", f"{category.title()} Skills" if category else "Agent Skills")

    # Repo type — language-based heuristics
    if node.type == "repo":
        if "obsidian" in name:
            return ("web_app", "nextjs", "Web Frontends")
        if "nanoGPT" in name or "gpt" in name or "llm" in name or "transformer" in name:
            return ("ml_ai", "pytorch", "ML / AI")
        if "fastapi" in name or "api" in name or "backend" in name:
            return ("api_service", "fastapi" if "fastapi" in name else "python_lib", "API Services")
        if "cli" in name:
            return ("cli_tool", "rust_cli" if "rust" in name else "python_lib", "CLI Tools")
        if "docs" in name or "documentation" in name or "wiki" in name:
            return ("documentation", "docs_site", "Documentation")
        if "docker" in name or "k8s" in name or "infra" in name:
            return ("infrastructure", "", "Infrastructure")
        if "test" in name:
            return ("other", "", "Testing")

        # Language-based
        lang_set = set(languages)
        if "TypeScript" in lang_set and "JavaScript" in lang_set:
            return ("web_app", "nextjs" if "next" in name else "react", "Web Frontends")
        if "Python" in lang_set:
            if any(kw in name for kw in ["ml", "train", "model", "torch", "keras"]):
                return ("ml_ai", "pytorch", "ML / AI")
            return ("cli_tool", "python_lib", "CLI Tools")
        if "Rust" in lang_set:
            return ("cli_tool", "rust_cli", "CLI Tools")
        if "Go" in lang_set:
            return ("api_service", "go_service", "API Services")

        return ("other", "", "Miscellaneous")

    return ("other", "", "Other")


def classify(model: GraphModel) -> GraphModel:
    """
    Classify all nodes in the model using LLM (with heuristic fallback).

    Batches nodes in groups of ~30 to avoid LLM output limits.
    Sets category, subcategory (subtype), and cluster for each node.
    Also infers edges based on shared clusters and other relationships.

    Returns:
        GraphModel with classifications applied.
    """
    nodes = list(model.nodes.values())

    # ── Try LLM with batching ──
    BATCH_SIZE = 30
    classifications: dict[str, dict] = {}

    if nodes:
        for i in range(0, len(nodes), BATCH_SIZE):
            batch = nodes[i:i + BATCH_SIZE]
            prompt = _build_classification_prompt(batch)
            llm_result = _call_llm(prompt)

            if llm_result:
                try:
                    data = json.loads(llm_result)
                    for item in data:
                        node_id = item.get("id")
                        if node_id and node_id in model.nodes:
                            classifications[node_id] = item
                    logger.debug("Batch %d/%d: classified %d nodes",
                                 i // BATCH_SIZE + 1,
                                 (len(nodes) + BATCH_SIZE - 1) // BATCH_SIZE,
                                 len(data))
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning("Failed to parse LLM response as JSON (batch %d): %s",
                                   i // BATCH_SIZE + 1, e)
                    logger.debug("Raw response: %s", llm_result[:200] if llm_result else "None")

    # Apply classifications (LLM or heuristic fallback)
    for node in nodes:
        if node.id in classifications:
            cls = classifications[node.id]
            node.subtype = cls.get("subcategory", cls.get("subtype", ""))
            cluster_name = cls.get("cluster", "Other")
            model.assign_to_cluster(node.id, cluster_name)
        else:
            # Heuristic fallback
            category, subcategory, cluster_name = _heuristic_classify(node)
            node.subtype = subcategory
            model.assign_to_cluster(node.id, cluster_name)

    # ── Infer inter-repo relationships ──

    # Same remote host → same_remote edge
    remote_groups: dict[str, list[str]] = {}
    for node in nodes:
        remote = node.metadata.get("remote", "")
        if remote:
            host = _extract_host(remote)
            if host:
                remote_groups.setdefault(host, []).append(node.id)

    for host, ids in remote_groups.items():
        if len(ids) > 1:
            for i in range(len(ids) - 1):
                model.add_edge(ids[i], ids[i + 1], kind="same_remote",
                               label=host)

    # Same cluster → similar edge
    cluster_members: dict[str, list[str]] = {}
    for node in nodes:
        if node.cluster:
            cluster_members.setdefault(node.cluster, []).append(node.id)

    for cname, ids in cluster_members.items():
        if len(ids) > 1 and cname not in ("Other", "Miscellaneous"):
            for i in range(min(len(ids) - 1, 3)):  # Limit edges to avoid noise
                model.add_edge(ids[i], ids[i + 1], kind="similar",
                               label=cname)

    # Repo → skill: if name contains skill name or vice versa
    repos = [n for n in nodes if n.type == "repo"]
    skills = [n for n in nodes if n.type == "skill"]
    for repo in repos:
        for skill in skills:
            rname = repo.name.lower()
            sname = skill.name.lower()
            if (sname in rname or rname in sname) and len(sname) > 3:
                model.add_edge(repo.id, skill.id, kind="uses",
                               label="related skill")

    return model


def _extract_host(remote_url: str) -> str:
    """Extract hostname from git remote URL."""
    for prefix in ("https://", "git@", "ssh://"):
        if remote_url.startswith(prefix):
            url = remote_url[len(prefix):]
            return url.split("/")[0].split(":")[0]
    return ""
