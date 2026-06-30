"""Render the section<->article graph as an interactive HTML you can pan/zoom/search.

Examples
--------
  # the whole web of *connected* nodes (isolated sections hidden):
  python scripts/visualize_graph.py --book laxmikanth_6

  # focus on one chapter's sections and the articles they cite:
  python scripts/visualize_graph.py --chapter 26

  # everything, including sections with no article edges (shows how sparse it is):
  python scripts/visualize_graph.py --full

Opens to data/processed/<book>/graph.html (or --out PATH).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
from pyvis.network import Network

from upsc_rag.config import get_settings
from upsc_rag.indexing.graph_store import load_graph

_SECTION_COLOR = "#4f86c6"  # blue
_ARTICLE_COLOR = "#e0a458"  # amber


def _section_label(data: dict) -> str:
    path = data.get("section_path") or []
    return path[-1] if path else data.get("section_id", "?")


def _select_nodes(graph: nx.Graph, chapter: int | None, full: bool) -> list[str]:
    """Pick which nodes to draw: a chapter's neighbourhood, the connected core, or all."""
    if chapter is not None:
        sections = [
            n for n, d in graph.nodes(data=True)
            if d.get("kind") == "section" and d.get("chapter_num") == chapter
        ]
        articles = {a for s in sections for a in graph.neighbors(s)}
        return sections + list(articles)
    if full:
        return list(graph.nodes)
    # default: drop isolated nodes so the actual web is visible
    return [n for n in graph.nodes if graph.degree(n) > 0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize the section<->article graph")
    parser.add_argument("--book", default="laxmikanth_6")
    parser.add_argument("--chapter", type=int, default=None, help="Focus on one chapter number")
    parser.add_argument("--full", action="store_true", help="Include isolated (edge-less) nodes")
    parser.add_argument("--out", default=None, help="Output HTML path")
    args = parser.parse_args()

    settings = get_settings()
    processed = settings.resolve(settings.processed_dir) / args.book
    graph = load_graph(processed / "graph.pkl")

    nodes = _select_nodes(graph, args.chapter, args.full)
    sub = graph.subgraph(nodes)

    net = Network(height="850px", width="100%", bgcolor="#111418", font_color="#e8e8e8",
                  notebook=False, cdn_resources="in_line", directed=False)
    net.barnes_hut(gravity=-8000, spring_length=120)

    for node, data in sub.nodes(data=True):
        deg = sub.degree(node)
        if data.get("kind") == "article":
            net.add_node(node, label=data.get("article", "?"), color=_ARTICLE_COLOR,
                         shape="dot", size=8 + 2 * deg, title=f"{data.get('article')} — cited by {deg} section(s)")
        else:
            path = " > ".join(data.get("section_path") or [])
            net.add_node(node, label=_section_label(data), color=_SECTION_COLOR,
                         shape="box", size=10 + 2 * deg,
                         title=f"{path}\n(ch {data.get('chapter_num')}, p{data.get('page_start')}) — {deg} article(s)")

    for a, b, ed in sub.edges(data=True):
        net.add_edge(a, b, value=ed.get("weight", 1))

    out = Path(args.out) if args.out else processed / "graph.html"
    # Write UTF-8 ourselves: pyvis.write_html() uses the platform default (cp1252 on
    # Windows), which can't encode the book's unicode/mojibake characters.
    out.write_text(net.generate_html(notebook=False), encoding="utf-8")

    n_sec = sum(1 for _, d in sub.nodes(data=True) if d.get("kind") == "section")
    n_art = sum(1 for _, d in sub.nodes(data=True) if d.get("kind") == "article")
    print(f"Drew {n_sec} section + {n_art} article nodes, {sub.number_of_edges()} edges")
    print(f"Open: {out}")


if __name__ == "__main__":
    main()
