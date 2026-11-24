"""Tiny dependency-free linear algebra used by the Kalman filter.

Matrices are row-major lists of lists.  The tracker only needs small dense
matrices (4x4 at most), so clarity beats speed here.
"""

from __future__ import annotations

INF = float("inf")


def zeros(rows: int, cols: int) -> list[list[float]]:
    return [[0.0] * cols for _ in range(rows)]


def identity(n: int) -> list[list[float]]:
    out = zeros(n, n)
    for i in range(n):
        out[i][i] = 1.0
    return out


def mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    rows, inner, cols = len(a), len(b), len(b[0])
    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        ai = a[i]
        oi = out[i]
        for k in range(inner):
            aik = ai[k]
            if aik == 0.0:
                continue
            bk = b[k]
            for j in range(cols):
                oi[j] += aik * bk[j]
    return out


def mat_add(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def mat_sub(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[x - y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def transpose(a: list[list[float]]) -> list[list[float]]:
    return [list(col) for col in zip(*a)]


def mat_vec(a: list[list[float]], x: list[float]) -> list[float]:
    return [sum(aij * xj for aij, xj in zip(row, x)) for row in a]


def scalar_mul(a: list[list[float]], s: float) -> list[list[float]]:
    return [[v * s for v in row] for row in a]


def inverse(a: list[list[float]]) -> list[list[float]]:
    """Gauss-Jordan inverse with partial pivoting.  Raises on singular input."""
    n = len(a)
    aug = [list(row) + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("singular matrix")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        pv = aug[col][col]
        aug[col] = [v / pv for v in aug[col]]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor == 0.0:
                continue
            aug[r] = [v - factor * pv2 for v, pv2 in zip(aug[r], aug[col])]
    return [row[n:] for row in aug]


def det2(m: list[list[float]]) -> float:
    return m[0][0] * m[1][1] - m[0][1] * m[1][0]


def inv2(m: list[list[float]]) -> list[list[float]]:
    d = det2(m)
    if abs(d) < 1e-12:
        raise ValueError("singular 2x2 matrix")
    return [[m[1][1] / d, -m[0][1] / d], [-m[1][0] / d, m[0][0] / d]]


def quad_form2(y: list[float], s: list[list[float]]) -> float:
    si = inv2(s)
    return y[0] * (si[0][0] * y[0] + si[0][1] * y[1]) + y[1] * (si[1][0] * y[0] + si[1][1] * y[1])
