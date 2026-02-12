"""Tests for pre-white-balance feature across normalization, processor, CLI, and engine layers."""

import os

os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import argparse
import json

import numpy as np
import pytest

from negpy.features.exposure.normalization import (
    compute_pre_wb_offsets,
    apply_pre_white_balance,
    analyze_log_exposure_bounds,
)
from negpy.features.exposure.processor import NormalizationProcessor
from negpy.features.process.models import ProcessConfig, ProcessMode
from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig


@pytest.fixture(scope="session", autouse=True)
def qapp():
    """Override conftest.py qapp — these tests don't need Qt."""
    yield None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_biased_image(shape: tuple[int, int] = (100, 100)) -> np.ndarray:
    """Creates a linear RGB image with a deliberate green cast and real dynamic range.

    Uses a gradient base with green offset, so each channel has meaningful
    floor-to-ceil spread while the green channel is systematically brighter.
    """
    h, w = shape
    rng = np.random.RandomState(42)
    # Base gradient from 0.05 to 0.95 with noise
    base = np.linspace(0.1, 0.8, w, dtype=np.float32).reshape(1, w)
    base = np.tile(base, (h, 1))
    noise = rng.uniform(-0.05, 0.05, (h, w)).astype(np.float32)

    img = np.empty((h, w, 3), dtype=np.float32)
    img[:, :, 0] = np.clip(base + noise - 0.1, 0.02, 0.95)  # R dimmer
    img[:, :, 1] = np.clip(base + noise + 0.1, 0.02, 0.95)  # G brighter (cast)
    img[:, :, 2] = np.clip(base + noise - 0.1, 0.02, 0.95)  # B dimmer
    return img


def _make_neutral_image(shape: tuple[int, int] = (100, 100)) -> np.ndarray:
    """Creates a neutral image with dynamic range but all channels equal."""
    h, w = shape
    rng = np.random.RandomState(123)
    base = np.linspace(0.1, 0.8, w, dtype=np.float32).reshape(1, w)
    base = np.tile(base, (h, 1))
    noise = rng.uniform(-0.03, 0.03, (h, w)).astype(np.float32)
    ch = np.clip(base + noise, 0.02, 0.95)
    return np.stack([ch, ch, ch], axis=-1)


# ===========================================================================
# 1. compute_pre_wb_offsets
# ===========================================================================


class TestComputePreWBOffsets:
    def test_zero_strength_returns_zeros(self) -> None:
        """Strength=0 should always return (0, 0, 0) regardless of image content."""
        img = _make_biased_image()
        bounds = analyze_log_exposure_bounds(img)
        offsets = compute_pre_wb_offsets(img, bounds, strength=0.0)
        assert offsets == (0.0, 0.0, 0.0)

    def test_neutral_image_returns_near_zero_offsets(self) -> None:
        """An image with equal channels should produce negligible offsets."""
        img = _make_neutral_image()
        bounds = analyze_log_exposure_bounds(img)
        offsets = compute_pre_wb_offsets(img, bounds, strength=1.0)
        for o in offsets:
            assert abs(o) < 1e-4, f"Neutral image should have ~0 offset, got {o}"

    def test_biased_image_produces_nonzero_offsets(self) -> None:
        """An image with a green cast should produce non-zero offsets."""
        img = _make_biased_image()
        bounds = analyze_log_exposure_bounds(img)
        offsets = compute_pre_wb_offsets(img, bounds, strength=1.0)
        # At least one offset should be significantly non-zero
        assert any(abs(o) > 0.01 for o in offsets), f"Expected non-zero offsets for biased image, got {offsets}"

    def test_offsets_sum_to_approximately_zero(self) -> None:
        """Offsets shift channels toward a common mean, so they should roughly sum to zero."""
        img = _make_biased_image()
        bounds = analyze_log_exposure_bounds(img)
        offsets = compute_pre_wb_offsets(img, bounds, strength=1.0)
        assert abs(sum(offsets)) < 1e-4, f"Offsets should sum to ~0, got {sum(offsets)}"

    def test_green_channel_offset_differs_from_rb(self) -> None:
        """Green-biased image: G has a different brightness profile, so its offset
        should differ from R and B (which are equal)."""
        img = _make_biased_image()  # G brighter, R=B dimmer
        bounds = analyze_log_exposure_bounds(img)
        offsets = compute_pre_wb_offsets(img, bounds, strength=1.0)
        # R and B have the same bias, so their offsets should be nearly equal
        assert abs(offsets[0] - offsets[2]) < 1e-4, "R and B offsets should be nearly equal"
        # G offset should differ from R/B because of the channel imbalance
        assert abs(offsets[1] - offsets[0]) > 0.01, f"G offset should differ from R offset: G={offsets[1]:.4f}, R={offsets[0]:.4f}"

    def test_strength_scales_offsets_linearly(self) -> None:
        """Doubling the strength should approximately double the offsets."""
        img = _make_biased_image()
        bounds = analyze_log_exposure_bounds(img)
        off_half = compute_pre_wb_offsets(img, bounds, strength=0.5)
        off_full = compute_pre_wb_offsets(img, bounds, strength=1.0)
        for h, f in zip(off_half, off_full):
            assert abs(f - 2.0 * h) < 1e-6, f"Expected linear scaling: half={h}, full={f}"

    def test_with_roi(self) -> None:
        """ROI restricts analysis to a sub-region and produces valid 3-float offsets."""
        img = _make_biased_image((200, 200))
        bounds = analyze_log_exposure_bounds(img)
        roi = (50, 150, 50, 150)  # center crop
        offsets = compute_pre_wb_offsets(img, bounds, strength=1.0, roi=roi)
        assert len(offsets) == 3
        # Offsets should still sum to ~0 (neutral shift)
        assert abs(sum(offsets)) < 1e-4
        # Should be non-trivial for a biased image
        assert any(abs(o) > 0.01 for o in offsets)

    def test_with_analysis_buffer(self) -> None:
        """Analysis buffer excludes borders from the analysis."""
        img = _make_biased_image()
        bounds = analyze_log_exposure_bounds(img)
        offsets = compute_pre_wb_offsets(img, bounds, strength=1.0, analysis_buffer=0.1)
        # For uniform image, buffer shouldn't change results much
        offsets_no_buf = compute_pre_wb_offsets(img, bounds, strength=1.0, analysis_buffer=0.0)
        for a, b in zip(offsets, offsets_no_buf):
            assert abs(a - b) < 0.01

    def test_returns_three_floats(self) -> None:
        """Return type should be a tuple of 3 floats."""
        img = _make_biased_image()
        bounds = analyze_log_exposure_bounds(img)
        offsets = compute_pre_wb_offsets(img, bounds, strength=0.75)
        assert len(offsets) == 3
        for o in offsets:
            assert isinstance(o, float)

    def test_e6_mode_produces_valid_offsets(self) -> None:
        """Pre-WB should work for E6 (positive) mode too."""
        img = _make_biased_image()
        bounds = analyze_log_exposure_bounds(img, process_mode=ProcessMode.E6)
        offsets = compute_pre_wb_offsets(img, bounds, strength=1.0)
        assert len(offsets) == 3
        # Should still produce meaningful offsets for a biased image
        assert any(abs(o) > 0.001 for o in offsets)


# ===========================================================================
# 2. apply_pre_white_balance
# ===========================================================================


class TestApplyPreWhiteBalance:
    def test_zero_offsets_returns_unchanged(self) -> None:
        """Zero offsets should return the original image."""
        img = np.random.rand(50, 50, 3).astype(np.float32)
        result = apply_pre_white_balance(img, (0.0, 0.0, 0.0))
        np.testing.assert_array_equal(result, img)

    def test_positive_offset_reduces_channel(self) -> None:
        """A positive offset should reduce (darken) that channel."""
        img = np.full((10, 10, 3), 0.6, dtype=np.float32)
        result = apply_pre_white_balance(img, (0.1, 0.0, 0.0))
        assert result[0, 0, 0] < img[0, 0, 0], "Positive R offset should reduce R channel"
        np.testing.assert_allclose(result[0, 0, 0], 0.5, atol=1e-5)
        np.testing.assert_allclose(result[0, 0, 1], 0.6, atol=1e-5)  # G unchanged
        np.testing.assert_allclose(result[0, 0, 2], 0.6, atol=1e-5)  # B unchanged

    def test_negative_offset_increases_channel(self) -> None:
        """A negative offset should increase (brighten) that channel."""
        img = np.full((10, 10, 3), 0.4, dtype=np.float32)
        result = apply_pre_white_balance(img, (-0.1, 0.0, 0.0))
        assert result[0, 0, 0] > img[0, 0, 0], "Negative R offset should increase R channel"
        np.testing.assert_allclose(result[0, 0, 0], 0.5, atol=1e-5)

    def test_output_clamped_to_0_1(self) -> None:
        """Result should be clamped to [0, 1] range."""
        img = np.full((10, 10, 3), 0.1, dtype=np.float32)
        result = apply_pre_white_balance(img, (0.5, 0.5, 0.5))
        assert np.min(result) >= 0.0

        img2 = np.full((10, 10, 3), 0.9, dtype=np.float32)
        result2 = apply_pre_white_balance(img2, (-0.5, -0.5, -0.5))
        assert np.max(result2) <= 1.0

    def test_preserves_shape_and_dtype(self) -> None:
        """Output should have same shape and float32 dtype."""
        img = np.random.rand(64, 48, 3).astype(np.float32)
        result = apply_pre_white_balance(img, (0.05, -0.02, 0.03))
        assert result.shape == img.shape
        assert result.dtype == np.float32

    def test_per_channel_independence(self) -> None:
        """Each channel offset should only affect that channel."""
        img = np.full((10, 10, 3), 0.5, dtype=np.float32)
        result = apply_pre_white_balance(img, (0.1, 0.2, 0.3))
        np.testing.assert_allclose(result[0, 0, 0], 0.4, atol=1e-5)
        np.testing.assert_allclose(result[0, 0, 1], 0.3, atol=1e-5)
        np.testing.assert_allclose(result[0, 0, 2], 0.2, atol=1e-5)


# ===========================================================================
# 3. NormalizationProcessor integration
# ===========================================================================


class TestNormalizationProcessorPreWB:
    def _make_context(self, mode: str = ProcessMode.C41) -> PipelineContext:
        return PipelineContext(scale_factor=1.0, original_size=(100, 100), process_mode=mode)

    def test_pre_wb_zero_does_not_alter_output(self) -> None:
        """pre_wb=0 should produce identical output to default config."""
        img = _make_biased_image()
        config_off = ProcessConfig(pre_wb=0.0)
        config_default = ProcessConfig()

        ctx_off = self._make_context()
        ctx_default = self._make_context()

        res_off = NormalizationProcessor(config_off).process(img, ctx_off)
        res_default = NormalizationProcessor(config_default).process(img, ctx_default)

        np.testing.assert_array_equal(res_off, res_default)

    def test_pre_wb_active_changes_output(self) -> None:
        """pre_wb>0 on a biased image should produce a different result."""
        img = _make_biased_image()
        config_off = ProcessConfig(pre_wb=0.0)
        config_on = ProcessConfig(pre_wb=1.0)

        ctx_off = self._make_context()
        ctx_on = self._make_context()

        res_off = NormalizationProcessor(config_off).process(img, ctx_off)
        res_on = NormalizationProcessor(config_on).process(img, ctx_on)

        assert not np.array_equal(res_off, res_on), "pre_wb should alter the normalized output"

    def test_pre_wb_reduces_channel_spread(self) -> None:
        """After pre-WB, channel means should be closer together than without it."""
        img = _make_biased_image()
        config_off = ProcessConfig(pre_wb=0.0)
        config_on = ProcessConfig(pre_wb=1.0)

        res_off = NormalizationProcessor(config_off).process(img, self._make_context())
        res_on = NormalizationProcessor(config_on).process(img, self._make_context())

        means_off = [float(np.mean(res_off[:, :, ch])) for ch in range(3)]
        means_on = [float(np.mean(res_on[:, :, ch])) for ch in range(3)]

        spread_off = max(means_off) - min(means_off)
        spread_on = max(means_on) - min(means_on)

        assert spread_on < spread_off, f"Pre-WB should reduce channel spread: off={spread_off:.4f}, on={spread_on:.4f}"

    def test_pre_wb_stores_offsets_in_metrics(self) -> None:
        """When active, pre-WB offsets should be stored in context.metrics."""
        img = _make_biased_image()
        config = ProcessConfig(pre_wb=0.75)
        ctx = self._make_context()

        NormalizationProcessor(config).process(img, ctx)

        assert "pre_wb_offsets" in ctx.metrics
        offsets = ctx.metrics["pre_wb_offsets"]
        assert len(offsets) == 3

    def test_pre_wb_inactive_no_offsets_in_metrics(self) -> None:
        """When pre_wb=0, no offsets should be stored in metrics."""
        img = _make_neutral_image()
        config = ProcessConfig(pre_wb=0.0)
        ctx = self._make_context()

        NormalizationProcessor(config).process(img, ctx)

        assert "pre_wb_offsets" not in ctx.metrics

    def test_pre_wb_output_in_valid_range(self) -> None:
        """Result should always be in [0, 1]."""
        img = _make_biased_image()
        config = ProcessConfig(pre_wb=1.0)
        ctx = self._make_context()

        res = NormalizationProcessor(config).process(img, ctx)

        assert np.min(res) >= 0.0
        assert np.max(res) <= 1.0

    def test_pre_wb_with_locked_roll_average(self) -> None:
        """Pre-WB should work when using locked roll-average bounds."""
        img = _make_biased_image()
        bounds = analyze_log_exposure_bounds(img)
        config = ProcessConfig(
            pre_wb=1.0,
            use_roll_average=True,
            locked_floors=bounds.floors,
            locked_ceils=bounds.ceils,
        )
        ctx = self._make_context()

        res = NormalizationProcessor(config).process(img, ctx)

        assert np.min(res) >= 0.0
        assert np.max(res) <= 1.0
        assert "pre_wb_offsets" in ctx.metrics


# ===========================================================================
# 4. ProcessConfig model
# ===========================================================================


class TestProcessConfigPreWB:
    def test_default_pre_wb_is_zero(self) -> None:
        """Default pre_wb should be 0.0 for backward compatibility."""
        config = ProcessConfig()
        assert config.pre_wb == 0.0

    def test_pre_wb_set_via_constructor(self) -> None:
        config = ProcessConfig(pre_wb=0.75)
        assert config.pre_wb == 0.75

    def test_pre_wb_survives_serialization_roundtrip(self) -> None:
        """pre_wb should survive WorkspaceConfig serialization/deserialization."""
        original = WorkspaceConfig(
            process=ProcessConfig(pre_wb=0.85),
        )
        flat = original.to_dict()
        assert flat["pre_wb"] == 0.85

        restored = WorkspaceConfig.from_flat_dict(flat)
        assert restored.process.pre_wb == 0.85

    def test_pre_wb_missing_from_flat_dict_uses_default(self) -> None:
        """Old configs without pre_wb should deserialize with default=0.0."""
        data = {"process_mode": "C41", "density": 1.0}
        config = WorkspaceConfig.from_flat_dict(data)
        assert config.process.pre_wb == 0.0


# ===========================================================================
# 5. CLI --pre-wb flag
# ===========================================================================


class TestCLIPreWB:
    def test_parser_accepts_pre_wb_flag(self) -> None:
        from negpy.cli.batch import build_parser

        parser = build_parser()
        args = parser.parse_args(["--pre-wb", "0.75", "input.dng"])
        assert args.pre_wb == 0.75

    def test_parser_default_pre_wb_is_none(self) -> None:
        from negpy.cli.batch import build_parser

        parser = build_parser()
        args = parser.parse_args(["input.dng"])
        assert args.pre_wb is None

    def test_build_config_applies_pre_wb(self) -> None:
        from negpy.cli.batch import build_config

        args = argparse.Namespace(
            mode="c41",
            output_format="tiff",
            output="./export",
            color_space="adobe-rgb",
            density=None,
            grade=None,
            pre_wb=0.8,
            sharpen=None,
            dpi=None,
            print_size=None,
            preview=False,
            filename_pattern=None,
            no_gpu=False,
            settings=None,
            crop_offset=None,
            dust_remove=False,
            dust_threshold=None,
            dust_size=None,
            preset=None,
            inputs=["file.dng"],
        )
        config = build_config(args, {"cli": {}, "processing": {}})
        assert config.process.pre_wb == 0.8

    def test_build_config_pre_wb_none_preserves_default(self) -> None:
        from negpy.cli.batch import build_config

        args = argparse.Namespace(
            mode="c41",
            output_format="tiff",
            output="./export",
            color_space="adobe-rgb",
            density=None,
            grade=None,
            pre_wb=None,
            sharpen=None,
            dpi=None,
            print_size=None,
            preview=False,
            filename_pattern=None,
            no_gpu=False,
            settings=None,
            crop_offset=None,
            dust_remove=False,
            dust_threshold=None,
            dust_size=None,
            preset=None,
            inputs=["file.dng"],
        )
        config = build_config(args, {"cli": {}, "processing": {}})
        assert config.process.pre_wb == 0.0

    def test_build_config_pre_wb_from_user_config(self) -> None:
        """pre_wb from user config (~/.negpy/config.json) should be picked up."""
        from negpy.cli.batch import build_config

        args = argparse.Namespace(
            mode="c41",
            output_format="tiff",
            output="./export",
            color_space="adobe-rgb",
            density=None,
            grade=None,
            pre_wb=None,
            sharpen=None,
            dpi=None,
            print_size=None,
            preview=False,
            filename_pattern=None,
            no_gpu=False,
            settings=None,
            crop_offset=None,
            dust_remove=False,
            dust_threshold=None,
            dust_size=None,
            preset=None,
            inputs=["file.dng"],
        )
        user_config = {"cli": {}, "processing": {"pre_wb": 0.6}}
        config = build_config(args, user_config)
        assert config.process.pre_wb == 0.6

    def test_build_config_cli_flag_overrides_user_config(self) -> None:
        """CLI --pre-wb should override user config value."""
        from negpy.cli.batch import build_config

        args = argparse.Namespace(
            mode="c41",
            output_format="tiff",
            output="./export",
            color_space="adobe-rgb",
            density=None,
            grade=None,
            pre_wb=0.9,
            sharpen=None,
            dpi=None,
            print_size=None,
            preview=False,
            filename_pattern=None,
            no_gpu=False,
            settings=None,
            crop_offset=None,
            dust_remove=False,
            dust_threshold=None,
            dust_size=None,
            preset=None,
            inputs=["file.dng"],
        )
        user_config = {"cli": {}, "processing": {"pre_wb": 0.3}}
        config = build_config(args, user_config)
        assert config.process.pre_wb == 0.9

    def test_init_config_includes_pre_wb(self, tmp_path, monkeypatch) -> None:
        """--init-config should include pre_wb in the generated config."""
        from negpy.cli.batch import generate_default_config

        monkeypatch.setattr("negpy.cli.batch.CONFIG_DIR", str(tmp_path / ".negpy"))
        monkeypatch.setattr("negpy.cli.batch.CONFIG_FILE", str(tmp_path / ".negpy" / "config.json"))
        monkeypatch.setattr("negpy.cli.batch.PRESETS_DIR", str(tmp_path / ".negpy" / "presets"))

        generate_default_config()

        data = json.loads((tmp_path / ".negpy" / "config.json").read_text())
        assert "pre_wb" in data["processing"]
        assert data["processing"]["pre_wb"] == 0.0


# ===========================================================================
# 6. Engine cache invalidation
# ===========================================================================


class TestEngineCacheInvalidation:
    def test_pre_wb_change_invalidates_base_cache(self) -> None:
        """Changing pre_wb should cause a re-render (cache miss on base stage)."""
        from negpy.services.rendering.engine import DarkroomEngine

        engine = DarkroomEngine()
        img = np.random.rand(80, 80, 3).astype(np.float32)

        settings_off = WorkspaceConfig(process=ProcessConfig(pre_wb=0.0))
        settings_on = WorkspaceConfig(process=ProcessConfig(pre_wb=0.75))

        res_off = engine.process(img, settings_off, source_hash="test_pre_wb")
        assert engine.cache.base is not None
        base_hash_off = engine.cache.base.config_hash

        res_on = engine.process(img, settings_on, source_hash="test_pre_wb")
        base_hash_on = engine.cache.base.config_hash

        assert base_hash_off != base_hash_on, "pre_wb change should invalidate base cache"
        assert not np.array_equal(res_off, res_on), "Different pre_wb should produce different results"
