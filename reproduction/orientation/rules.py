"""Auditable collider orientation and conservative Meek R1--R4 closure.

There is no default variable ordering or control/response prior.  Separating
sets must be supplied by actual independence tests: a missing record is not an
empty separating set.  The caller can mark triples ambiguous after checking
additional separating sets, as in conservative PC.

Rules follow Meek (1995), https://arxiv.org/abs/1302.4972, and the R1--R4
formulation in Pearl's Causality, second edition, chapter 2, p. 51:
https://web.cs.ucla.edu/~kaoru/CAUSALITY-REPRINTED-W-CORRECTIONS-2013/
causality-ch2-june2013.pdf.  In particular, R4 explicitly requires the a--d
shield (adjacency of either orientation) in a--b, a--c, c->d->b.

Unlike plain graph-only Meek closure, R1, R3, and R4 require their relevant
unshielded noncollider to have a recorded, unambiguous separating set.  This
can leave more edges unresolved when evidence is incomplete.  R2 only uses a
directed chain and does not need a noncollider classification.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from numbers import Integral
from typing import Iterable, Mapping


def _pair(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _triple(a: int, center: int, b: int) -> tuple[int, int, int]:
    a, b = _pair(a, b)
    return a, center, b


def _strong_components(n_nodes: int, arrows: set[tuple[int, int]]) -> list[set[int]]:
    """Tarjan components; used to reject an entire simultaneous cycle."""
    children = [[] for _ in range(n_nodes)]
    for source, target in sorted(arrows):
        children[source].append(target)
    indices: dict[int, int] = {}
    low: dict[int, int] = {}
    stack: list[int] = []
    active: set[int] = set()
    components: list[set[int]] = []

    def visit(node: int) -> None:
        indices[node] = low[node] = len(indices)
        stack.append(node)
        active.add(node)
        for child in children[node]:
            if child not in indices:
                visit(child)
                low[node] = min(low[node], low[child])
            elif child in active:
                low[node] = min(low[node], indices[child])
        if low[node] == indices[node]:
            component = set()
            while True:
                item = stack.pop()
                active.remove(item)
                component.add(item)
                if item == node:
                    break
            components.append(component)

    for node in range(n_nodes):
        if node not in indices:
            visit(node)
    return components


def orient_pdag(
    n_nodes: int,
    skeleton_edges: Iterable[tuple[int, int]],
    separating_sets: Mapping[tuple[int, int], Iterable[int]],
    *,
    background_arrows: Iterable[tuple[int, int]] = (),
    ambiguous_triples: Iterable[tuple[int, int, int]] = (),
    apply_meek: bool = True,
) -> dict:
    """Orient a skeleton from data, then optionally add background knowledge.

    Nodes are integers ``0 .. n_nodes-1``.  Pair keys may be in either order;
    conflicting duplicate separator records are rejected.  An explicit empty
    set asserts a tested marginal independence; omitted keys remain unknown.
    Ambiguous triples are ``(endpoint, center, endpoint)`` and override the
    single separating-set classification for that particular triple.

    All returned values are JSON serializable.  ``stage=data`` means the arrow
    was obtained before any background knowledge; every arrow added afterward,
    including subsequent Meek propagation, has ``stage=background``.  Competing
    simultaneous proposals, directed cycles, and violations of known
    noncolliders are reported rather than resolved by traversal order.  Rejected
    undirected edges stay blocked for the rest of this invocation.  Background
    knowledge never overwrites a data arrow or invents a missing skeleton edge.

    ``acyclic`` only concerns directed cycles, and does not certify that the
    partial graph is extendable or that statistical/causal assumptions hold.
    Set ``apply_meek=False`` to retain collider orientation and direct background
    arrows while skipping both data and background Meek propagation.
    """
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, Integral) or n_nodes < 0:
        raise ValueError("n_nodes must be a nonnegative integer")
    n_nodes = int(n_nodes)
    if not isinstance(apply_meek, bool):
        raise ValueError("apply_meek must be boolean")

    def node(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError("node indices must be integers")
        value = int(value)
        if not 0 <= value < n_nodes:
            raise ValueError("node index is outside 0 .. n_nodes-1")
        return value

    def edge(raw: Iterable[int]) -> tuple[int, int]:
        values = tuple(raw)
        if len(values) != 2:
            raise ValueError("an edge must have two endpoints")
        a, b = map(node, values)
        if a == b:
            raise ValueError("self edges are not permitted")
        return a, b

    skeleton = {_pair(*edge(raw)) for raw in skeleton_edges}
    separators: dict[tuple[int, int], frozenset[int]] = {}
    for raw_pair, raw_set in separating_sets.items():
        key = _pair(*edge(raw_pair))
        if key in skeleton:
            raise ValueError("a separating-set record cannot refer to a retained edge")
        separator = frozenset(node(value) for value in raw_set)
        if set(key) & separator:
            raise ValueError("a separating set cannot contain either tested endpoint")
        if key in separators and separators[key] != separator:
            raise ValueError("duplicate pair has inconsistent separating sets")
        separators[key] = separator
    background = sorted({edge(raw) for raw in background_arrows})
    ambiguous = set()
    for raw in ambiguous_triples:
        values = tuple(raw)
        if len(values) != 3:
            raise ValueError("an ambiguous triple must contain three nodes")
        a, center, b = map(node, values)
        if len({a, center, b}) != 3:
            raise ValueError("an ambiguous triple needs three distinct nodes")
        if (
            _pair(a, center) not in skeleton
            or _pair(b, center) not in skeleton
            or _pair(a, b) in skeleton
        ):
            raise ValueError("an ambiguous triple must be unshielded in the skeleton")
        ambiguous.add(_triple(a, center, b))

    neighbours = [set() for _ in range(n_nodes)]
    for a, b in skeleton:
        neighbours[a].add(b)
        neighbours[b].add(a)
    triples = []
    classification = {}
    for center in range(n_nodes):
        for a, b in combinations(sorted(neighbours[center]), 2):
            if _pair(a, b) in skeleton:
                continue
            key = _triple(a, center, b)
            separator = separators.get(_pair(a, b))
            if key in ambiguous:
                status = "ambiguous"
            elif separator is None:
                status = "unresolved"
            elif center in separator:
                status = "noncollider"
            else:
                status = "collider"
            classification[key] = status
            triples.append(
                {
                    "endpoints": [a, b],
                    "center": center,
                    "status": status,
                    "separating_set": None if separator is None else sorted(separator),
                }
            )

    directed: set[tuple[int, int]] = set()
    undirected = set(skeleton)
    blocked: set[tuple[int, int]] = set()
    orientation_log = []
    conflicts = []

    def noncollider(a: int, center: int, b: int) -> bool:
        return classification.get(_triple(a, center, b)) == "noncollider"

    def proposal(source: int, target: int, rule: str, evidence: dict) -> dict:
        return {"source": source, "target": target, "rule": rule, "evidence": evidence}

    def apply(proposals: list[dict], stage: str) -> bool:
        grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for item in proposals:
            source, target = item["source"], item["target"]
            key = _pair(source, target)
            if key not in skeleton:
                conflicts.append({**item, "stage": stage, "type": "missing_skeleton_edge"})
            elif (target, source) in directed:
                conflicts.append(
                    {
                        **item,
                        "stage": stage,
                        "type": "opposite_existing_arrow",
                        "existing_arrow": [target, source],
                    }
                )
            elif key in blocked:
                conflicts.append({**item, "stage": stage, "type": "blocked_by_previous_conflict"})
            elif (source, target) not in directed and key not in blocked:
                grouped[source, target].append(item)
        candidates = set(grouped)
        opposed = {arrow for arrow in candidates if arrow[::-1] in candidates}
        for a, b in sorted({_pair(*arrow) for arrow in opposed}):
            blocked.add((a, b))
            conflicts.append(
                {
                    "type": "opposing_proposals",
                    "stage": stage,
                    "edge": [a, b],
                    "proposals": grouped[a, b] + grouped[b, a],
                }
            )
        candidates -= opposed

        # Treat a simultaneous batch symmetrically.  Reject both new arrows of
        # a prohibited collider instead of favoring the first in node order.
        prospective = directed | candidates
        collider_rejections = set()
        for a, center, b in sorted(classification):
            if (
                classification[a, center, b] == "noncollider"
                and (a, center) in prospective
                and (b, center) in prospective
            ):
                rejected = {(a, center), (b, center)} & candidates
                if rejected:
                    collider_rejections |= rejected
                    conflicts.append(
                        {
                            "type": "noncollider_violation",
                            "stage": stage,
                            "triple": [a, center, b],
                            "separating_set": sorted(separators[_pair(a, b)]),
                            "rejected_arrows": [list(item) for item in sorted(rejected)],
                        }
                    )
        candidates -= collider_rejections
        blocked.update(_pair(*arrow) for arrow in collider_rejections)

        prospective = directed | candidates
        cycle_rejections = set()
        for component in _strong_components(n_nodes, prospective):
            if len(component) > 1:
                rejected = {
                    arrow for arrow in candidates if arrow[0] in component and arrow[1] in component
                }
                cycle_rejections |= rejected
                conflicts.append(
                    {
                        "type": "directed_cycle",
                        "stage": stage,
                        "nodes": sorted(component),
                        "rejected_arrows": [list(item) for item in sorted(rejected)],
                    }
                )
        candidates -= cycle_rejections
        blocked.update(_pair(*arrow) for arrow in cycle_rejections)
        for source, target in sorted(candidates):
            directed.add((source, target))
            undirected.remove(_pair(source, target))
            supports = sorted(grouped[source, target], key=lambda item: item["rule"])
            orientation_log.append(
                {**supports[0], "stage": stage, "supporting_proposals": supports}
            )
        return bool(candidates)

    collider_proposals = []
    for item in triples:
        if item["status"] == "collider":
            a, b = item["endpoints"]
            center = item["center"]
            evidence = {"triple": [a, center, b], "separating_set": item["separating_set"]}
            collider_proposals.extend(
                [
                    proposal(a, center, "collider", evidence),
                    proposal(b, center, "collider", evidence),
                ]
            )
    apply(collider_proposals, "data")

    def meek_closure(stage: str) -> None:
        while True:
            proposals = []
            for a, b in sorted(undirected - blocked):
                for source, target in ((a, b), (b, a)):
                    # R1: c -> source -- target, c not adjacent to target.
                    for c in sorted(neighbours[source] - neighbours[target] - {target}):
                        if (c, source) in directed and noncollider(c, source, target):
                            proposals.append(
                                proposal(
                                    source,
                                    target,
                                    "R1",
                                    {
                                        "triple": [c, source, target],
                                        "arrows": [[c, source]],
                                        "separating_set": sorted(separators[_pair(c, target)]),
                                    },
                                )
                            )
                    # R2: source -> c -> target; the reverse would form a cycle.
                    for c in sorted(neighbours[source] & neighbours[target]):
                        if (source, c) in directed and (c, target) in directed:
                            proposals.append(
                                proposal(source, target, "R2", {"chain": [source, c, target]})
                            )
                    # R3: source--c, source--d, c->target<-d, c not adjacent d.
                    parents = sorted(
                        c
                        for c in neighbours[source] & neighbours[target]
                        if _pair(source, c) in undirected and (c, target) in directed
                    )
                    for c, d in combinations(parents, 2):
                        if _pair(c, d) not in skeleton and noncollider(c, source, d):
                            proposals.append(
                                proposal(
                                    source,
                                    target,
                                    "R3",
                                    {
                                        "noncollider": [c, source, d],
                                        "arrows": [[c, target], [d, target]],
                                        "separating_set": sorted(separators[_pair(c, d)]),
                                    },
                                )
                            )
                    # R4: source--c, c->d->target, c not adjacent target;
                    # source and d must be adjacent (the shield).
                    for c in sorted(neighbours[source] - neighbours[target] - {target}):
                        if _pair(source, c) not in undirected or not noncollider(c, source, target):
                            continue
                        for d in sorted(neighbours[source] & neighbours[c] & neighbours[target]):
                            if (c, d) in directed and (d, target) in directed:
                                proposals.append(
                                    proposal(
                                        source,
                                        target,
                                        "R4",
                                        {
                                            "noncollider": [c, source, target],
                                            "chain": [c, d, target],
                                            "shield": list(_pair(source, d)),
                                            "separating_set": sorted(separators[_pair(c, target)]),
                                        },
                                    )
                                )
            if not apply(proposals, stage):
                break

    if apply_meek:
        meek_closure("data")
    if background:
        apply(
            [
                proposal(source, target, "background", {"requested_arrow": [source, target]})
                for source, target in background
            ],
            "background",
        )
        if apply_meek:
            meek_closure("background")
    acyclic = all(len(component) == 1 for component in _strong_components(n_nodes, directed))
    status = "conflicted" if conflicts else "partially_directed" if undirected else "fully_directed"
    return {
        "directed_edges": [list(item) for item in sorted(directed)],
        "undirected_edges": [list(item) for item in sorted(undirected)],
        "orientation_log": orientation_log,
        "unshielded_triples": triples,
        "conflicts": conflicts,
        "acyclic": acyclic,
        "status": status,
        "blocked_edges": [list(item) for item in sorted(blocked)],
    }
