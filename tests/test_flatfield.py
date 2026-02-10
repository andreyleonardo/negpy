"""Tests for negpy.features.flatfield.logic — flat-field correction."""

import os

os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import numpy as np
import pytest

from negpy.features.flatfield.logic import normalize_flatfield, apply_flatfield


@pytest.fixture(scope="session", autouse=True)
def qapp():
    """Override conftest.py qapp — these tests don't need Qt."""
    yield None


class TestNormalizeFlatfield:
    def test_uniform_returns_ones(self):
        """A uniform-value image should produce a map of all ~1.0."""
        flat = np.full((100, 100, 3), 0.5, dtype=np.float32)
        result = normalize_flatfield(flat)
        np.testing.assert_allclose(result, 1.0, atol=1e-5)

    def test_vignette_pattern_mean_is_one(self):
        """After normalization, per-channel mean should be ~1.0."""
        flat = np.ones((200, 200, 3), dtype=np.float32) * 0.8
        # Make center brighter
        flat[50:150, 50:150, :] = 1.0
        result = normalize_flatfield(flat)
        for ch in range(3):
            assert abs(result[:, :, ch].mean() - 1.0) < 1e-4

    def test_vignette_center_greater_than_edges(self):
        """Center-bright flat should produce values >1.0 at center, <1.0 at edges."""
        flat = np.ones((200, 200, 3), dtype=np.float32) * 0.5
        flat[80:120, 80:120, :] = 1.0
        result = normalize_flatfield(flat)
        center_val = result[100, 100, 0]
        corner_val = result[0, 0, 0]
        assert center_val > 1.0
        assert corner_val < 1.0

    def test_clamps_minimum(self):
        """Flat field with zero pixels should clamp to epsilon, not produce inf/nan."""
        flat = np.zeros((50, 50, 3), dtype=np.float32)
        result = normalize_flatfield(flat)
        assert not np.any(np.isinf(result))
        assert not np.any(np.isnan(result))
        assert np.all(result > 0)

    def test_preserves_shape(self):
        """Output should have same shape as input."""
        flat = np.random.rand(120, 80, 3).astype(np.float32)
        result = normalize_flatfield(flat)
        assert result.shape == flat.shape

    def test_output_dtype_float32(self):
        flat = np.ones((50, 50, 3), dtype=np.float32) * 0.7
        result = normalize_flatfield(flat)
        assert result.dtype == np.float32


class TestApplyFlatfield:
    def test_identity_correction(self):
        """Dividing by all-1.0 map returns the original image unchanged."""
        image = np.random.rand(100, 100, 3).astype(np.float32)
        flat = np.ones((100, 100, 3), dtype=np.float32)
        result = apply_flatfield(image, flat)
        np.testing.assert_allclose(result, image, atol=1e-6)

    def test_shape_mismatch_resizes(self):
        """Flat field of different size gets resized to match image."""
        image = np.full((200, 300, 3), 0.5, dtype=np.float32)
        flat = np.ones((100, 150, 3), dtype=np.float32)
        result = apply_flatfield(image, flat)
        assert result.shape == image.shape

    def test_corrects_vignette(self):
        """Applying correction to a vignetted image should make it more uniform."""
        # Create a uniform scene
        scene = np.full((200, 200, 3), 0.6, dtype=np.float32)
        # Create a vignette pattern (dim at edges)
        vignette = np.ones((200, 200, 3), dtype=np.float32)
        for y in range(200):
            for x in range(200):
                dist = np.sqrt((y - 100) ** 2 + (x - 100) ** 2) / 141.0
                vignette[y, x, :] = max(0.5, 1.0 - 0.5 * dist)
        # Apply vignette to scene (simulating what the scanner does)
        vignetted = scene * vignette
        # Normalize the flat and correct
        flat_norm = normalize_flatfield(vignette.copy())
        corrected = apply_flatfield(vignetted, flat_norm)
        # Corrected image should be more uniform than vignetted
        std_before = np.std(vignetted[:, :, 0])
        std_after = np.std(corrected[:, :, 0])
        assert std_after < std_before

    def test_output_clipped(self):
        """Result is clipped to [0.0, 1.0]."""
        image = np.full((50, 50, 3), 0.9, dtype=np.float32)
        flat = np.full((50, 50, 3), 0.3, dtype=np.float32)  # would produce 3.0
        result = apply_flatfield(image, flat)
        assert np.all(result <= 1.0)
        assert np.all(result >= 0.0)

    def test_preserves_dtype(self):
        image = np.random.rand(50, 50, 3).astype(np.float32)
        flat = np.ones((50, 50, 3), dtype=np.float32)
        result = apply_flatfield(image, flat)
        assert result.dtype == np.float32
