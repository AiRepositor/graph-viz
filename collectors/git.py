#!/usr/bin/env python3
"""
Git repo collector — discovers local git repos and extracts metadata.
"""

from __future__ import annotations

import logging
import subprocess
import os
from pathlib import Path
from graph_model import Node, GraphModel

logger = logging.getLogger(__name__)


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
    except (PermissionError, OSError):
        pass
    return total


def _has_changes(repo_path: Path) -> bool:
    try:
        r = subprocess.run(
            ["git", "-C", str(repo_path), "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        return bool(r.stdout.strip())
    except (subprocess.TimeoutExpired, OSError):
        return False


def _get_languages(repo_path: Path) -> list[str]:
    """Detect primary languages from file extensions in the repo root."""
    langs: dict[str, int] = {}
    ext_map = {
        ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
        ".tsx": "TypeScript", ".jsx": "JavaScript", ".rs": "Rust",
        ".go": "Go", ".java": "Java", ".rb": "Ruby",
        ".c": "C", ".cpp": "C++", ".h": "C", ".hpp": "C++",
        ".md": "Markdown", ".yaml": "YAML", ".yml": "YAML",
        ".json": "JSON", ".toml": "TOML", ".sh": "Shell",
        ".css": "CSS", ".html": "HTML", ".swift": "Swift",
        ".kt": "Kotlin", ".ex": "Elixir", ".exs": "Elixir",
        ".vue": "Vue", ".svelte": "Svelte",
    }
    try:
        for f in repo_path.iterdir():
            if f.is_file():
                ext = f.suffix.lower()
                if ext in ext_map:
                    lang = ext_map[ext]
                    langs[lang] = langs.get(lang, 0) + 1
    except PermissionError:
        pass
    sorted_langs = sorted(langs.items(), key=lambda x: -x[1])
    return [lang for lang, _ in sorted_langs[:3]]


def _get_repo_node(repo_path: Path) -> Node | None:
    """Extract metadata from a single git repo and return a Node."""
    try:
        name = repo_path.name

        remote = subprocess.run(
            ["git", "-C", str(repo_path), "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=5
        ).stdout.strip()

        branch = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5
        ).stdout.strip() or "unknown"

        last_commit = subprocess.run(
            ["git", "-C", str(repo_path), "log", "-1",
             "--format=%H|%h|%s|%ar", "--date=relative"],
            capture_output=True, text=True, timeout=5
        ).stdout.strip()

        # Parse last commit
        commit_hash, commit_short, commit_msg, commit_time = "","","",""
        if last_commit:
            parts = last_commit.split("|", 3)
            if len(parts) == 4:
                commit_hash, commit_short, commit_msg, commit_time = parts

        size_kb = _dir_size(repo_path / ".git") // 1024
        has_changes = _has_changes(repo_path)
        languages = _get_languages(repo_path)

        return Node.repo(
            name=name,
            path=str(repo_path),
            remote=remote,
            branch=branch,
            commit_hash=commit_short or commit_hash[:8] if commit_hash else "",
            commit_msg=commit_msg,
            commit_time=commit_time,
            size_kb=size_kb,
            has_uncommitted=has_changes,
            languages=languages,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.debug("Skipping %s: %s", repo_path, e)
        return None


def collect(root_dirs: list[str] | None = None,
            max_depth: int = 4,
            model: GraphModel | None = None) -> GraphModel:
    """
    Scan directories for git repos and add them to the model.

    Args:
        root_dirs: Directories to scan (default: ~ and ~/projects if exists)
        max_depth: Max subdirectory depth to search
        model: Existing GraphModel to populate (creates new one if None)

    Returns:
        GraphModel with repo nodes added.
    """
    if root_dirs is None:
        home = Path.home()
        root_dirs = [str(home)]
        # Add common project dirs if they exist
        for extra in ["projects", "work", "dev", "src"]:
            p = home / extra
            if p.is_dir():
                root_dirs.append(str(p))

    if model is None:
        model = GraphModel()

    seen: set[Path] = set()

    for root in root_dirs:
        root_p = Path(root).expanduser().resolve()
        if not root_p.is_dir():
            continue

        try:
            # Walk depth-limited using rglob with a path-depth check
            for candidate in root_p.rglob(".git"):
                if candidate.is_dir():
                    repo_path = candidate.parent
                    if repo_path in seen:
                        continue
                    seen.add(repo_path)

                    # Depth check: how deep from root is this?
                    try:
                        rel = repo_path.relative_to(root_p)
                        depth = len(rel.parts)
                    except ValueError:
                        depth = 0

                    if depth > max_depth:
                        continue

                    node = _get_repo_node(repo_path)
                    if node:
                        model.add_node(node)
        except PermissionError:
            continue

    return model
