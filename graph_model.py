#!/usr/bin/env python3
"""
Abstract Graph Model — intermediate representation decoupling data from rendering.

Defines:
  - Node: a typed entity (repo, skill, profile, config, etc.)
  - Edge: a relationship between two nodes
  - Cluster: a labeled group of nodes (e.g. "Frontend Apps", "ML Models")
  - GraphModel: the full container with layout hints, stats, and serialisation
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any


def _safe_id(raw: str) -> str:
    """Convert a string to a valid Graphviz DOT identifier.

    Replaces hyphens and other non-alphanumeric chars with underscores
    to avoid DOT port-parsing issues.
    """
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", raw)
    safe = re.sub(r"_+", "_", safe)       # collapse multiple underscores
    safe = safe.strip("_")
    return safe or "node"


# ── Node types ──────────────────────────────────────────────────────────────

NODE_TYPES = {
    "repo":        "Git repository",
    "skill":       "Hermes agent skill",
    "profile":     "Hermes agent profile",
    "config":      "Configuration file",
    "cron":        "Cron job / scheduled task",
    "plugin":      "Hermes plugin",
    "tool":        "Hermes tool module",
    "session":     "Conversation session",
    "file":        "Generic file artifact",
    "unknown":     "Unclassified",
}


@dataclass
class Node:
    """A single node in the graph."""

    id: str                           # Unique identifier (e.g. "repo:obsidian-clone")
    name: str                         # Display name
    type: str = "unknown"             # One of NODE_TYPES keys
    subtype: str = ""                 # Optional finer-grained type (e.g. "nextjs", "ml")
    metadata: dict[str, Any] = field(default_factory=dict)
    cluster: str = ""                 # Cluster/group name (set by intelligence layer)

    @classmethod
    def repo(cls, name: str, path: str, **meta) -> "Node":
        return cls(
            id=_safe_id(f"repo:{name}"),
            name=name,
            type="repo",
            metadata={"path": path, **meta},
        )

    @classmethod
    def skill(cls, name: str, category: str = "", **meta) -> "Node":
        return cls(
            id=_safe_id(f"skill:{name}"),
            name=name,
            type="skill",
            metadata={"category": category, **meta},
        )

    @classmethod
    def profile(cls, name: str) -> "Node":
        return cls(id=_safe_id(f"profile:{name}"), name=name, type="profile")
    
    @classmethod
    def config(cls, name: str = "config.yaml") -> "Node":
        return cls(id="config", name=name, type="config")
    
    @classmethod
    def cron(cls, count: int) -> "Node":
        return cls(id="cron", name=f"{count} jobs", type="cron",
                   metadata={"count": count})

    @classmethod
    def plugin(cls, name: str) -> "Node":
        return cls(id=_safe_id(f"plugin:{name}"), name=name, type="plugin")

    @classmethod
    def tool(cls, name: str) -> "Node":
        return cls(id=_safe_id(f"tool:{name}"), name=name, type="tool")

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return f"[{self.type}:{self.name}]"


# ── Edge types ────────────────────────────────────────────────────────────

EDGE_STYLES = {
    "depends_on":    {"label": "depends on",    "color": "#e74c3c", "style": "solid"},
    "relates_to":    {"label": "relates to",    "color": "#3498db", "style": "dashed"},
    "contains":      {"label": "contains",      "color": "#2ecc71", "style": "solid"},
    "fork_of":       {"label": "fork of",       "color": "#f39c12", "style": "dotted"},
    "same_remote":   {"label": "same remote",   "color": "#9b59b6", "style": "dashed"},
    "same_tech":     {"label": "same tech",     "color": "#1abc9c", "style": "dotted"},
    "uses":          {"label": "uses",          "color": "#3498db", "style": "solid"},
    "similar":       {"label": "similar",       "color": "#95a5a6", "style": "dotted"},
    "owned_by":      {"label": "owned by",      "color": "#e67e22", "style": "dashed"},
}


@dataclass
class Edge:
    """A directed relationship between two nodes."""

    source: str        # Source node id
    target: str        # Target node id
    kind: str = "relates_to"
    label: str = ""
    weight: float = 1.0

    def get_style(self) -> dict:
        return EDGE_STYLES.get(self.kind, EDGE_STYLES["relates_to"])

    def to_dict(self) -> dict:
        st = self.get_style()
        return {
            "source": self.source,
            "target": self.target,
            "kind": self.kind,
            "label": self.label or st["label"],
            "color": st["color"],
            "style": st["style"],
            "weight": self.weight,
        }


# ── Cluster ──────────────────────────────────────────────────────────────

@dataclass
class Cluster:
    """A group of related nodes, rendered as a named box/subgraph."""

    id: str
    name: str
    description: str = ""
    color: str = "#2c3e50"
    node_ids: set[str] = field(default_factory=set)

    def add_node(self, node_id: str) -> None:
        self.node_ids.add(node_id)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "color": self.color,
            "node_ids": sorted(self.node_ids),
        }


# ── Full Graph Model ────────────────────────────────────────────────────

_CLUSTER_PALETTE = [
    "#1a1a2e", "#16213e", "#0f3460", "#533483",
    "#2d2d2d", "#1a3a2e", "#3a1a1a", "#2a2a1a",
    "#1a2a3a", "#2a1a2a", "#1a3a3a", "#3a2a1a",
]


class GraphModel:
    """
    The intermediate graph representation.

    Collectors populate nodes and edges; the intelligence layer assigns
    clusters and hints; renderers consume it to produce output.
    """

    def __init__(self, title: str = "Agent Project Graph"):
        self.title = title
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.clusters: dict[str, Cluster] = {}
        self._palette_idx = 0

    # ── Node management ──

    def add_node(self, node: Node) -> "GraphModel":
        self.nodes[node.id] = node
        return self

    def add_nodes(self, nodes: list[Node]) -> "GraphModel":
        for n in nodes:
            self.add_node(n)
        return self

    def get_node(self, node_id: str) -> Node | None:
        return self.nodes.get(node_id)

    # ── Edge management ──

    def add_edge(self, source: str, target: str,
                 kind: str = "relates_to", label: str = "",
                 weight: float = 1.0) -> "GraphModel":
        # Only add if both nodes exist and edge is not duplicate
        if source not in self.nodes or target not in self.nodes:
            return self
        for e in self.edges:
            if e.source == source and e.target == target and e.kind == kind:
                return self
        self.edges.append(Edge(source, target, kind, label, weight))
        return self

    # ── Cluster management ──

    def ensure_cluster(self, name: str, description: str = "",
                       color: str | None = None) -> Cluster:
        existing = self.clusters.get(name)
        if existing:
            return existing
        if color is None:
            color = _CLUSTER_PALETTE[self._palette_idx % len(_CLUSTER_PALETTE)]
            self._palette_idx += 1
        cluster = Cluster(
            id=f"cluster:{name.lower().replace(' ', '_')}",
            name=name,
            description=description,
            color=color,
        )
        self.clusters[name] = cluster
        return cluster

    def assign_to_cluster(self, node_id: str, cluster_name: str) -> None:
        node = self.nodes.get(node_id)
        if node:
            node.cluster = cluster_name
            cluster = self.ensure_cluster(cluster_name)
            cluster.add_node(node_id)

    # ── Layout hints ──

    @property
    def suggested_engine(self) -> str:
        """Pick a layout engine based on graph characteristics."""
        n = len(self.nodes)
        if n <= 8:
            return "dot"
        elif n <= 20:
            return "neato"
        elif n <= 50:
            return "fdp"
        else:
            return "sfdp"

    # ── Stats ──

    @property
    def stats(self) -> dict:
        type_counts: dict[str, int] = {}
        for node in self.nodes.values():
            type_counts[node.type] = type_counts.get(node.type, 0) + 1
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "total_clusters": len(self.clusters),
            "types": type_counts,
            "suggested_engine": self.suggested_engine,
        }

    # ── Serialisation ──

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "stats": self.stats,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "clusters": [c.to_dict() for c in self.clusters.values()],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "GraphModel":
        g = cls(title=data.get("title", "Graph"))
        for nd in data.get("nodes", []):
            n = Node(**nd)
            g.nodes[n.id] = n
        for ed in data.get("edges", []):
            g.edges.append(Edge(**{k: v for k, v in ed.items()
                                   if k in ("source", "target", "kind", "label", "weight")}))
        for cd in data.get("clusters", []):
            c = Cluster(**{k: v for k, v in cd.items()
                          if k in ("id", "name", "description", "color")})
            c.node_ids = set(cd.get("node_ids", []))
            g.clusters[c.name] = c
        return g
