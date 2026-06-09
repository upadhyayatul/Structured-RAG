"""Knowledge graph over the book's sections and the Constitutional Articles they cite.

A bipartite graph with two node kinds:

  * ``section``  — one per TOC section (keyed by the parent chunk id). Carries the
                   hierarchy metadata so retrieval can build a citation from a node.
  * ``article``  — one per distinct Article reference (e.g. ``Article 361``).

Sections link to the articles they mention (``MENTIONS`` edges, weighted by how
often). Walking ``section -> article -> section`` therefore surfaces *other*
sections that discuss the same Articles — the cross-reference expansion that plain
vector search misses. Edges are derived for free from each chunk's ``entities`` and
``parent_id``; no LLM call is involved.
"""
from __future__ import annotations

import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

# Node-id helpers keep the two namespaces from colliding (a section id could in
# theory equal an article string).
def _section_node(section_id: str) -> str:
    return f"section::{section_id}"


def _article_node(article: str) -> str:
    return f"article::{article}"


def _section_id_of(chunk: dict[str, Any]) -> str | None:
    """The section a chunk belongs to: itself if a parent, else its parent_id."""
    if chunk.get("content_type") == "parent":
        return chunk.get("id")
    return chunk.get("parent_id")


_SECTION_META_FIELDS = (
    "section_path",
    "chapter_title",
    "chapter_num",
    "part",
    "page_start",
    "page_end",
)


def build_graph(chunks: Iterable[dict[str, Any]]) -> nx.Graph:
    """Build the section<->article graph from chunk dicts (parents + children)."""
    graph = nx.Graph()

    # Aggregate per section: its metadata (preferring the parent chunk) and a
    # Counter of how often each article is mentioned across the section's chunks.
    section_meta: dict[str, dict[str, Any]] = {}
    section_articles: dict[str, Counter] = defaultdict(Counter)

    for chunk in chunks:
        section_id = _section_id_of(chunk)
        if not section_id:
            continue
        # Parent chunk metadata wins; otherwise seed from the first child seen.
        if chunk.get("content_type") == "parent" or section_id not in section_meta:
            section_meta[section_id] = {f: chunk.get(f) for f in _SECTION_META_FIELDS}
        for article in chunk.get("entities") or []:
            if article:
                section_articles[section_id][article] += 1

    for section_id, meta in section_meta.items():
        graph.add_node(_section_node(section_id), kind="section", section_id=section_id, **meta)

    for section_id, articles in section_articles.items():
        s_node = _section_node(section_id)
        for article, count in articles.items():
            a_node = _article_node(article)
            if a_node not in graph:
                graph.add_node(a_node, kind="article", article=article)
            graph.add_edge(s_node, a_node, weight=count)

    return graph


def _article_idf(graph: nx.Graph, a_node: str, n_sections: int) -> float:
    """Inverse document frequency of an article: rare articles score high, ubiquitous ones low.

    df = number of sections mentioning the article. Article 14 (33 sections) gets a
    small weight; Article 361 (a handful) gets a large one — so a shared *rare*
    article dominates the neighbour ranking over many shared common ones.
    """
    df = graph.degree(a_node)
    return math.log((n_sections + 1) / (df + 1)) + 1.0


def expand_sections(
    graph: nx.Graph,
    seed_section_ids: Iterable[str],
    max_neighbors: int = 3,
    min_score: float = 0.0,
) -> list[str]:
    """Return section ids related to the seeds via shared Articles (1 hop through articles).

    Neighbours are scored by the summed IDF of the Articles they share with the seed
    set — sharing a *rare* article counts far more than sharing a ubiquitous one
    (Article 14/32), which keeps article-dense "hub" sections from dominating.
    Seeds themselves are excluded. Returns at most ``max_neighbors`` section ids,
    strongest first.
    """
    seeds = {_section_node(sid) for sid in seed_section_ids if _section_node(sid) in graph}
    if not seeds:
        return []

    n_sections = sum(1 for _, d in graph.nodes(data=True) if d.get("kind") == "section")

    candidate_score: Counter = Counter()   # neighbour section node -> summed IDF of shared articles
    for s_node in seeds:
        for a_node in graph.neighbors(s_node):
            idf = _article_idf(graph, a_node, n_sections)
            for nbr in graph.neighbors(a_node):
                if nbr in seeds or nbr == s_node:
                    continue
                candidate_score[nbr] += idf

    ranked = sorted(
        (n for n, score in candidate_score.items() if score > min_score),
        key=lambda n: candidate_score[n],
        reverse=True,
    )
    return [graph.nodes[n]["section_id"] for n in ranked[:max_neighbors]]


def section_metadata(graph: nx.Graph, section_id: str) -> dict[str, Any]:
    """Return the stored hierarchy metadata for a section id (empty dict if absent)."""
    node = _section_node(section_id)
    if node not in graph:
        return {}
    data = dict(graph.nodes[node])
    data.pop("kind", None)
    return data


def save_graph(graph: nx.Graph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(graph, f)


def load_graph(path: Path) -> nx.Graph:
    with path.open("rb") as f:
        return pickle.load(f)
