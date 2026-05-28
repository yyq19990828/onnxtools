"""Unit tests for the hand-rolled Kalman filters."""

from __future__ import annotations

import numpy as np
import pytest

from onnxtools.tracking.kalman import KalmanFilterXYAH, KalmanFilterXYSR

# ---------------------------------------------------------------------------
# KalmanFilterXYAH
# ---------------------------------------------------------------------------


class TestKalmanFilterXYAH:
    def test_initiate_shapes(self):
        kf = KalmanFilterXYAH()
        mean, cov = kf.initiate(np.array([100.0, 100.0, 0.5, 200.0]))
        assert mean.shape == (8,)
        assert cov.shape == (8, 8)
        # Velocity components start at zero.
        assert np.allclose(mean[4:], 0.0)
        # Diagonal positive.
        assert np.all(np.diag(cov) > 0)

    def test_predict_advances_by_velocity(self):
        kf = KalmanFilterXYAH()
        mean, cov = kf.initiate(np.array([100.0, 100.0, 0.5, 200.0]))
        # Inject velocity so we can verify F propagates it.
        mean[4] = 5.0  # vx
        mean[5] = -3.0  # vy
        new_mean, _ = kf.predict(mean, cov)
        assert pytest.approx(new_mean[0], abs=1e-4) == 105.0
        assert pytest.approx(new_mean[1], abs=1e-4) == 97.0

    def test_multi_predict_matches_predict_loop(self):
        kf = KalmanFilterXYAH()
        means = []
        covs = []
        for h in (100.0, 200.0, 50.0):
            m, c = kf.initiate(np.array([10.0, 20.0, 0.4, h]))
            m[4] = 1.0
            means.append(m)
            covs.append(c)
        means_arr = np.stack(means)
        covs_arr = np.stack(covs)

        loop_means = []
        loop_covs = []
        for m, c in zip(means, covs):
            nm, nc = kf.predict(m, c)
            loop_means.append(nm)
            loop_covs.append(nc)

        batch_means, batch_covs = kf.multi_predict(means_arr, covs_arr)
        np.testing.assert_allclose(batch_means, np.stack(loop_means), atol=1e-4)
        np.testing.assert_allclose(batch_covs, np.stack(loop_covs), atol=1e-3)

    def test_update_reduces_covariance(self):
        kf = KalmanFilterXYAH()
        mean, cov = kf.initiate(np.array([10.0, 20.0, 0.5, 100.0]))
        det_before = np.linalg.det(cov[:4, :4])
        new_mean, new_cov = kf.update(mean, cov, np.array([10.5, 20.5, 0.51, 101.0]))
        det_after = np.linalg.det(new_cov[:4, :4])
        # Measurement should shrink uncertainty.
        assert det_after < det_before
        # Diagonal must stay positive (PD).
        assert np.all(np.diag(new_cov) > 0)

    def test_gating_distance_nonnegative(self):
        kf = KalmanFilterXYAH()
        mean, cov = kf.initiate(np.array([10.0, 20.0, 0.5, 100.0]))
        d = kf.gating_distance(mean, cov, np.array([[11.0, 21.0, 0.51, 101.0], [50.0, 60.0, 0.6, 120.0]]))
        assert d.shape == (2,)
        assert np.all(d >= 0)
        # Farther measurement should have larger distance.
        assert d[1] > d[0]

    def test_empty_multi_predict(self):
        kf = KalmanFilterXYAH()
        means = np.zeros((0, 8), dtype=np.float32)
        covs = np.zeros((0, 8, 8), dtype=np.float32)
        m, c = kf.multi_predict(means, covs)
        assert m.shape == (0, 8)
        assert c.shape == (0, 8, 8)


# ---------------------------------------------------------------------------
# KalmanFilterXYSR
# ---------------------------------------------------------------------------


class TestKalmanFilterXYSR:
    def test_initiate_shapes(self):
        kf = KalmanFilterXYSR()
        mean, cov = kf.initiate(np.array([100.0, 100.0, 200.0 * 100.0, 2.0]))
        assert mean.shape == (7,)
        assert cov.shape == (7, 7)
        assert np.allclose(mean[4:], 0.0)

    def test_multi_predict_zeros_runaway_vs(self):
        """If vs would push area non-positive, multi_predict zeros it."""
        kf = KalmanFilterXYSR()
        mean, cov = kf.initiate(np.array([10.0, 20.0, 1000.0, 1.5]))
        mean[6] = -5000.0  # runaway vs
        means = mean[None, :]
        covs = cov[None, ...]
        new_means, _ = kf.multi_predict(means, covs)
        # vs reset to 0 so subsequent predict steps don't compound.
        assert float(new_means[0, 6]) == 0.0

    def test_update_reduces_covariance(self):
        kf = KalmanFilterXYSR()
        mean, cov = kf.initiate(np.array([10.0, 20.0, 5000.0, 1.5]))
        det_before = np.linalg.det(cov[:4, :4])
        new_mean, new_cov = kf.update(mean, cov, np.array([10.2, 20.2, 5050.0, 1.49]))
        det_after = np.linalg.det(new_cov[:4, :4])
        assert det_after < det_before
        assert np.all(np.diag(new_cov) >= -1e-6)

    def test_multi_predict_matches_predict_loop(self):
        kf = KalmanFilterXYSR()
        means = []
        covs = []
        for s in (1000.0, 5000.0, 200.0):
            m, c = kf.initiate(np.array([10.0, 20.0, s, 1.0]))
            m[4] = 2.0
            means.append(m)
            covs.append(c)
        means_arr = np.stack(means)
        covs_arr = np.stack(covs)

        loop_means = [kf.predict(m, c)[0] for m, c in zip(means, covs)]
        batch_means, _ = kf.multi_predict(means_arr, covs_arr)
        np.testing.assert_allclose(batch_means, np.stack(loop_means), atol=1e-4)
