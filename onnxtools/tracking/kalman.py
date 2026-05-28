"""Kalman filters for 2D bounding-box trackers.

Two state parameterisations are provided:

* :class:`KalmanFilterXYAH` — 8-D ``[cx, cy, a, h, vx, vy, va, vh]`` used by
  ByteTrack. Observation noise scales with target height (taller boxes ->
  looser noise), matching the original SORT/DeepSORT recipe.
* :class:`KalmanFilterXYSR` — 7-D ``[x, y, s, r, vx, vy, vs]`` used by
  OC-SORT. ``s`` is box area, ``r`` is aspect ratio (assumed constant — no
  velocity for r).

Both classes are pure-numpy with a vectorised :meth:`multi_predict` for hot
loops. Storage is ``float32`` for memory locality on edge devices; numerical
robustness during innovation update is handled by a small Tikhonov damping
term and a ``solve`` fallback when Cholesky factorisation fails.
"""

from __future__ import annotations

import numpy as np

# Chi-square 95% inverse for DoF 1..9 — used by gating_distance().
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}


# ---------------------------------------------------------------------------
# ByteTrack KF — 8D state [cx, cy, a, h, vx, vy, va, vh]
# ---------------------------------------------------------------------------


class KalmanFilterXYAH:
    """Kalman filter for bbox tracking in image space (ByteTrack flavour).

    State: ``[cx, cy, a, h, vx, vy, va, vh]`` — centre, aspect, height, plus
    derivatives. Observation: ``[cx, cy, a, h]``.
    """

    ndim = 4

    def __init__(self) -> None:
        dt = 1.0
        # State transition: position += velocity * dt
        self._motion_mat = np.eye(2 * self.ndim, dtype=np.float32)
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = dt
        self._update_mat = np.eye(self.ndim, 2 * self.ndim, dtype=np.float32)

        # Noise std weights — relative to current measurement (height).
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Initialise state from a single measurement ``[cx, cy, a, h]``."""
        m = np.asarray(measurement, dtype=np.float32)
        mean_pos = m
        mean_vel = np.zeros_like(m)
        mean = np.concatenate([mean_pos, mean_vel])

        h = m[3]
        std = np.array(
            [
                2 * self._std_weight_position * h,
                2 * self._std_weight_position * h,
                1e-2,
                2 * self._std_weight_position * h,
                10 * self._std_weight_velocity * h,
                10 * self._std_weight_velocity * h,
                1e-5,
                10 * self._std_weight_velocity * h,
            ],
            dtype=np.float32,
        )
        cov = np.diag(std * std).astype(np.float32)
        return mean, cov

    def _motion_cov(self, h: float) -> np.ndarray:
        std = np.array(
            [
                self._std_weight_position * h,
                self._std_weight_position * h,
                1e-2,
                self._std_weight_position * h,
                self._std_weight_velocity * h,
                self._std_weight_velocity * h,
                1e-5,
                self._std_weight_velocity * h,
            ],
            dtype=np.float32,
        )
        return np.diag(std * std).astype(np.float32)

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        motion_cov = self._motion_cov(mean[3])
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean.astype(np.float32), covariance.astype(np.float32)

    def multi_predict(self, means: np.ndarray, covariances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised batch predict over N tracks.

        Args:
            means: ``[N, 8]`` stacked state vectors.
            covariances: ``[N, 8, 8]`` stacked covariances.

        Returns:
            Updated ``(means, covariances)`` with the same shapes.
        """
        if means.shape[0] == 0:
            return means, covariances

        F = self._motion_mat
        # mean: [N, 8] · F.T -> [N, 8]
        new_means = means @ F.T

        # Batched motion covariance built from each track's height.
        hs = means[:, 3]
        n = means.shape[0]
        motion_cov = np.zeros((n, 8, 8), dtype=np.float32)
        diag = np.empty((n, 8), dtype=np.float32)
        diag[:, 0] = self._std_weight_position * hs
        diag[:, 1] = self._std_weight_position * hs
        diag[:, 2] = 1e-2
        diag[:, 3] = self._std_weight_position * hs
        diag[:, 4] = self._std_weight_velocity * hs
        diag[:, 5] = self._std_weight_velocity * hs
        diag[:, 6] = 1e-5
        diag[:, 7] = self._std_weight_velocity * hs
        diag *= diag  # variance = std^2
        idx = np.arange(8)
        motion_cov[:, idx, idx] = diag

        # F · Cov · F.T for each track (single einsum).
        new_covs = np.einsum("ij,njk,lk->nil", F, covariances, F) + motion_cov
        return new_means.astype(np.float32), new_covs.astype(np.float32)

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = np.array(
            [
                self._std_weight_position * h,
                self._std_weight_position * h,
                1e-1,
                self._std_weight_position * h,
            ],
            dtype=np.float32,
        )
        innovation_cov = np.diag(std * std).astype(np.float32)
        H = self._update_mat
        mean_proj = H @ mean
        cov_proj = H @ covariance @ H.T + innovation_cov
        return mean_proj.astype(np.float32), cov_proj.astype(np.float32)

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        projected_mean, projected_cov = self.project(mean, covariance)

        # Kalman gain: K = Cov · H.T · S^{-1}
        H = self._update_mat
        cov_HT = covariance @ H.T  # [8, 4]
        try:
            L = np.linalg.cholesky(projected_cov)
            tmp = np.linalg.solve(L, cov_HT.T)
            kalman_gain = np.linalg.solve(L.T, tmp).T
        except np.linalg.LinAlgError:
            damped = projected_cov + 1e-6 * np.eye(self.ndim, dtype=np.float32)
            kalman_gain = np.linalg.solve(damped.T, cov_HT.T).T

        innovation = np.asarray(measurement, dtype=np.float32) - projected_mean
        new_mean = mean + kalman_gain @ innovation
        new_cov = covariance - kalman_gain @ H @ covariance
        return new_mean.astype(np.float32), new_cov.astype(np.float32)

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
        metric: str = "maha",
    ) -> np.ndarray:
        mean_proj, cov_proj = self.project(mean, covariance)
        if only_position:
            mean_proj = mean_proj[:2]
            cov_proj = cov_proj[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean_proj
        if metric == "gaussian":
            return np.sum(d * d, axis=1).astype(np.float32)
        # Mahalanobis squared.
        try:
            L = np.linalg.cholesky(cov_proj)
            z = np.linalg.solve(L, d.T)
            return np.sum(z * z, axis=0).astype(np.float32)
        except np.linalg.LinAlgError:
            damped = cov_proj + 1e-6 * np.eye(cov_proj.shape[0], dtype=np.float32)
            inv = np.linalg.inv(damped)
            return np.einsum("ni,ij,nj->n", d, inv, d).astype(np.float32)


# ---------------------------------------------------------------------------
# OC-SORT KF — 7D state [x, y, s, r, vx, vy, vs]
# ---------------------------------------------------------------------------


class KalmanFilterXYSR:
    """Kalman filter for bbox tracking with constant-velocity area model.

    State: ``[cx, cy, s, r, vx, vy, vs]`` where ``s`` is area and ``r`` is
    aspect ratio (no velocity). Used by OC-SORT / SORT.
    """

    ndim = 4  # observation dim
    state_dim = 7

    def __init__(self) -> None:
        # F: state transition with dt=1 on (x, y, s) -> (vx, vy, vs)
        F = np.eye(7, dtype=np.float32)
        F[0, 4] = 1.0
        F[1, 5] = 1.0
        F[2, 6] = 1.0
        self._motion_mat = F

        # H: observe [x, y, s, r]
        H = np.zeros((4, 7), dtype=np.float32)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        H[3, 3] = 1.0
        self._update_mat = H

        # Process / measurement noise — match SORT defaults (rescaled to f32).
        self._R = np.diag([1.0, 1.0, 10.0, 10.0]).astype(np.float32)
        Q = np.eye(7, dtype=np.float32)
        Q[-1, -1] *= 0.01
        Q[4:, 4:] *= 0.01
        self._Q = Q

        P = np.eye(7, dtype=np.float32) * 10.0
        P[4:, 4:] *= 1000.0  # high uncertainty on initial velocities
        self._P0 = P

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``measurement`` is ``[cx, cy, s, r]``."""
        m = np.asarray(measurement, dtype=np.float32)
        mean = np.zeros(7, dtype=np.float32)
        mean[:4] = m
        return mean, self._P0.copy()

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        F = self._motion_mat
        new_mean = F @ mean
        new_cov = F @ covariance @ F.T + self._Q
        # Guard: if predicted area would go non-positive, zero velocity on s.
        if new_mean[2] + new_mean[6] <= 0:
            new_mean[6] = 0.0
        return new_mean.astype(np.float32), new_cov.astype(np.float32)

    def multi_predict(self, means: np.ndarray, covariances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if means.shape[0] == 0:
            return means, covariances
        F = self._motion_mat
        new_means = means @ F.T
        # Reset vs where area would go non-positive.
        bad = (new_means[:, 2] + new_means[:, 6]) <= 0
        if bad.any():
            new_means[bad, 6] = 0.0
        new_covs = np.einsum("ij,njk,lk->nil", F, covariances, F) + self._Q
        return new_means.astype(np.float32), new_covs.astype(np.float32)

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        H = self._update_mat
        S = H @ covariance @ H.T + self._R
        try:
            L = np.linalg.cholesky(S)
            tmp = np.linalg.solve(L, (covariance @ H.T).T)
            K = np.linalg.solve(L.T, tmp).T
        except np.linalg.LinAlgError:
            damped = S + 1e-6 * np.eye(self.ndim, dtype=np.float32)
            K = np.linalg.solve(damped.T, (covariance @ H.T).T).T
        z = np.asarray(measurement, dtype=np.float32)
        y = z - H @ mean
        new_mean = mean + K @ y
        new_cov = (np.eye(7, dtype=np.float32) - K @ H) @ covariance
        return new_mean.astype(np.float32), new_cov.astype(np.float32)
