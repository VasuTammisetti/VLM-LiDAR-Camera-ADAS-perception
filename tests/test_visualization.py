"""Tests for visualization module — no GPU required."""
import numpy as np
import os
import tempfile
import pytest
from src.visualization import load_velodyne, load_calib, project_lidar_to_image


class TestLoadVelodyne:
    def test_shape(self):
        """Velodyne loader should return Nx3 array."""
        dummy = np.random.rand(200, 4).astype(np.float32)
        path = os.path.join(tempfile.gettempdir(), "test_velo.bin")
        dummy.tofile(path)

        result = load_velodyne(path)
        assert result.shape == (200, 3)
        os.remove(path)

    def test_drops_reflectance(self):
        """Should drop the 4th column (reflectance)."""
        dummy = np.array([[1, 2, 3, 0.5], [4, 5, 6, 0.9]], dtype=np.float32)
        path = os.path.join(tempfile.gettempdir(), "test_velo2.bin")
        dummy.tofile(path)

        result = load_velodyne(path)
        assert result.shape[1] == 3
        np.testing.assert_array_almost_equal(result[0], [1, 2, 3])
        os.remove(path)


class TestLoadCalib:
    def _create_dummy_calib(self):
        path = os.path.join(tempfile.gettempdir(), "test_calib.txt")
        with open(path, 'w') as f:
            f.write("P0: " + " ".join(["0.0"] * 12) + "\n")
            f.write("P1: " + " ".join(["0.0"] * 12) + "\n")
            f.write("P2: " + " ".join(["1.0"] * 12) + "\n")
            f.write("P3: " + " ".join(["0.0"] * 12) + "\n")
            f.write("R0_rect: " + " ".join(["1.0"] * 9) + "\n")
            f.write("Tr_velo_to_cam: " + " ".join(["1.0"] * 12) + "\n")
            f.write("Tr_imu_to_velo: " + " ".join(["0.0"] * 12) + "\n")
        return path

    def test_shapes(self):
        path = self._create_dummy_calib()
        P2, R0, Tr = load_calib(path)
        assert P2.shape == (3, 4)
        assert R0.shape == (4, 4)
        assert Tr.shape == (4, 4)
        os.remove(path)

    def test_r0_is_4x4(self):
        """R0_rect should be embedded in 4x4 identity."""
        path = self._create_dummy_calib()
        _, R0, _ = load_calib(path)
        assert R0[3, 3] == 1.0
        os.remove(path)


class TestProjection:
    def test_filters_behind_camera(self):
        """Points behind camera (negative depth) should be removed."""
        points = np.array([[0, 0, 5], [0, 0, -5]], dtype=np.float64)
        P2 = np.eye(3, 4)
        R0 = np.eye(4)
        Tr = np.eye(4)
        u, v, d = project_lidar_to_image(points, P2, R0, Tr, 100, 100)
        assert len(d) == 1
        assert d[0] == 5.0