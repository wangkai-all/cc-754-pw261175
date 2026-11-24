"""Hungarian assignment algorithm, implemented from scratch.

Solves the rectangular minimum-cost assignment problem for ``n`` rows and
``m >= n`` columns; every row is assigned to a distinct column.  Costs may be
``inf`` to forbid a pair.  Returns ``None`` if no finite assignment exists.
"""

from __future__ import annotations

INF = float("inf")


def hungarian(cost: list[list[float]]) -> list[int] | None:
    if not cost:
        return []
    n = len(cost)
    m = len(cost[0])
    if n > m:
        raise ValueError("hungarian expects rows <= columns")
    if any(len(row) != m for row in cost):
        raise ValueError("ragged cost matrix")

    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    big = 1e15

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [big] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = big
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            if delta >= big / 2:
                return None
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * n
    for j in range(1, m + 1):
        row = p[j]
        if row:
            assignment[row - 1] = j - 1
    if any(a < 0 or a >= m for a in assignment):
        return None
    if any(cost[i][assignment[i]] >= INF for i in range(n)):
        return None
    return assignment


def assignment_cost(cost: list[list[float]], assignment: list[int]) -> float:
    total = 0.0
    for i, j in enumerate(assignment):
        total += cost[i][j]
    return total
