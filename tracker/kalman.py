"""Constant-velocity Kalman filter in 2-D, from scratch.

State: [x, y, vx, vy].  Measurement: [x, y].
"""

from __future__ import annotations

import math

from .linalg import inv2, det2, mat_add, mat_mul, mat_vec, transpose, zeros, identity, quad_form2

MEAS_DIM = 2
STATE_DIM = 4


def _outer(y: list[float], h: list[list[float]]) -> list[list[float]]:
    # outer product of a 2-vector and 2x4 H (= 4x2 * 2x2 style helpers inlined)
    out = zeros(STATE_DIM, MEAS_DIM)
    for i in range(STATE_DIM):
        for j in range(MEAS_DIM):
            out[i][j] = h[i][j] * 1.0
    return out


class KalmanTrack:
    def __init__(self, x: float, y: float, pos_var: float = 4.0, vel_var: float = 1.0):
        self.mean = [x, y, 0.0, 0.0]
        self.cov = [
            [pos_var, 0.0, 0.0, 0.0],
            [0.0, pos_var, 0.0, 0.0],
            [0.0, 0.0, vel_var, 0.0],
            [0.0, 0.0, 0.0, vel_var],
        ]

    def clone(self) -> "KalmanTrack":
        out = KalmanTrack.__new__(KalmanTrack)
        out.mean = list(self.mean)
        out.cov = [list(row) for row in self.cov]
        return out

    def predict(self, dt: float = 1.0, q_pos: float = 0.12, q_vel: float = 0.05) -> None:
        f = [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        # Discrete white-noise acceleration model.
        q = [
            [q_pos * dt**3 / 3.0, 0.0, q_pos * dt**2 / 2.0, 0.0],
            [0.0, q_pos * dt**3 / 3.0, 0.0, q_pos * dt**2 / 2.0],
            [q_pos * dt**2 / 2.0, 0.0, q_pos * dt, 0.0],
            [0.0, q_pos * dt**2 / 2.0, 0.0, q_pos * dt],
        ]
        self.mean = mat_vec(f, self.mean)
        self.cov = mat_add(mat_mul(mat_mul(f, self.cov), transpose(f)), q)

    def innovation(self, zx: float, zy: float, meas_var: float) -> tuple[float, list[float], list[list[float]]]:
        y = [zx - self.mean[0], zy - self.mean[1]]
        # S = H P H^T + R, H = [I 0]
        s = [
            [self.cov[0][0] + meas_var, self.cov[0][1]],
            [self.cov[1][0], self.cov[1][1] + meas_var],
        ]
        return quad_form2(y, s), y, s

    def log_likelihood(self, zx: float, zy: float, meas_var: float) -> tuple[float, float]:
        d2, _y, s = self.innovation(zx, zy, meas_var)
        det = det2(s)
        # -log N(z; Hx, S) = 0.5*d2 + 0.5*log(2*pi*|S|)
        cost = 0.5 * d2 + 0.5 * (math.log(2.0 * math.pi) + math.log(max(det, 1e-12)))
        return cost, d2

    def update(self, zx: float, zy: float, meas_var: float) -> None:
        _d2, y, s = self.innovation(zx, zy, meas_var)
        si = inv2(s)
        # P H^T is the first 4x2 block of P
        pht = [[self.cov[i][0], self.cov[i][1]] for i in range(STATE_DIM)]
        gain = [[sum(pht[i][k] * si[k][j] for k in range(2)) for j in range(2)] for i in range(STATE_DIM)]
        delta = mat_vec(gain, y)
        for i in range(STATE_DIM):
            self.mean[i] += delta[i]
        # H P is the first two rows of P
        hp = [self.cov[0][:], self.cov[1][:]]
        new_cov = []
        for i in range(STATE_DIM):
            row = []
            for j in range(STATE_DIM):
                row.append(self.cov[i][j] - sum(gain[i][k] * hp[k][j] for k in range(2)))
            new_cov.append(row)
        self.cov = new_cov
        # keep covariance symmetric against round-off
        for i in range(STATE_DIM):
            for j in range(i + 1, STATE_DIM):
                avg = 0.5 * (self.cov[i][j] + self.cov[j][i])
                self.cov[i][j] = self.cov[j][i] = avg

    @property
    def position(self) -> tuple[float, float]:
        return self.mean[0], self.mean[1]

    @property
    def velocity(self) -> tuple[float, float]:
        return self.mean[2], self.mean[3]
