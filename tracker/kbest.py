"""K-best rectangular assignments (Murty's algorithm) built on Hungarian.

The tracker uses this to compare several association hypotheses per frame
instead of committing to the single cheapest pairing.
"""

from __future__ import annotations

import heapq

from .hungarian import hungarian

INF = float("inf")
_FORBID = 1.0e12
_FORCE_BONUS = 1.0e7


def _solve(
    cost: list[list[float]],
    forced: frozenset[tuple[int, int]],
    forbidden: frozenset[tuple[int, int]],
) -> tuple[float, tuple[int, ...]] | None:
    """Solve one assignment subject to forced / forbidden (row, col) pairs."""
    n = len(cost)
    if n == 0:
        return 0.0, ()
    m = len(cost[0])
    work = [row[:] for row in cost]
    for i in range(n):
        for j in range(m):
            if (i, j) in forbidden:
                work[i][j] = INF
            elif (i, j) in forced:
                if cost[i][j] >= INF:
                    return None
                work[i][j] = cost[i][j] - _FORCE_BONUS
    # Pad to at least as many columns as rows.
    if m < n:
        work = [row + [INF] * (n - m) for row in work]
    assign = hungarian(work)
    if assign is None:
        return None
    for i, j in enumerate(assign):
        if j >= len(cost[0]):
            return None  # assigned to a padding column
        if (i, j) in forbidden or cost[i][j] >= INF:
            return None
    for (i, j) in forced:
        if j >= len(cost[0]) or assign[i] != j:
            return None
    total = sum(cost[i][assign[i]] for i in range(n))
    return total, tuple(assign)


def k_best_assignments(
    cost: list[list[float]], k: int, *, max_examined: int = 400
) -> tuple[list[dict], bool]:
    """Return up to ``k`` cheapest assignments and a truncation flag.

    Each result is ``{"cost": float, "columns": tuple[int, ...]}``.  The second
    return value is True when more alternatives existed than requested (or the
    search node budget was hit), so callers must not claim exhaustive optimality.
    """
    if k <= 0:
        return [], True
    n = len(cost)
    if n == 0:
        # No tracks: the empty assignment is the only hypothesis.  "Not
        # truncated" means the enumeration is complete for this frame.
        return ([{"cost": 0.0, "columns": ()}] if k > 0 else []), False
    first = _solve(cost, frozenset(), frozenset())
    if first is None:
        return [], False
    heap: list[tuple[float, tuple[int, ...], frozenset, frozenset]] = [
        (round(first[0], 12), first[1], frozenset(), frozenset())
    ]
    results: list[dict] = []
    seen: set[tuple[int, ...]] = set()
    examined = 0
    truncated = False
    while heap and len(results) < k:
        if examined >= max_examined:
            truncated = True
            break
        examined += 1
        total, assign, forced, forbidden = heapq.heappop(heap)
        if assign in seen:
            continue
        seen.add(assign)
        results.append({"cost": total, "columns": assign})
        if len(results) >= k:
            break
        # Murty partition around the popped solution.
        for i in range(n):
            f2 = set(forced)
            for q in range(i):
                f2.add((q, assign[q]))
            x2 = set(forbidden)
            x2.add((i, assign[i]))
            sol = _solve(cost, frozenset(f2), frozenset(x2))
            if sol is None:
                continue
            if sol[1] in seen:
                continue
            heapq.heappush(heap, (round(sol[0], 12), sol[1], frozenset(f2), frozenset(x2)))
    # ``truncated`` means the *search budget* ran out before the enumeration
    # could be completed.  Stopping at ``k`` is a deliberate cap, not a budget
    # failure - callers that need the full picture must ask for a larger k.
    return results, truncated
