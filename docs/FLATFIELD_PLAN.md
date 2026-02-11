# Plan: Flat-Field / Vignetting Correction for CLI

## Context

The user scans film negatives and some conversions show light leaking on the sides from the scanner backlight. This is a classic flat-field / vignetting issue: the backlight isn't perfectly uniform, so edges and corners get different illumination than the center.

NegPy has **no existing flat-field correction**. The closest features are `analysis_buffer` (excludes borders from normalization analysis) and CLAHE (adaptive local contrast) — neither actually corrects the per-pixel illumination.

The standard fix is to divide each scan by a reference "flat-field" frame (a blank scan capturing the backlight profile). This must happen in **linear space before log conversion**, making it a new pipeline stage between Geometry and Normalization.

### Scope

This plan is **CLI-only** — we add a `--flat-field` flag to the batch CLI. No GUI changes. The correction logic lives in its own module so the GUI can integrate it later.

## Pipeline Insertion Point

Current CPU pipeline in `DarkroomEngine.process()` (`negpy/services/rendering/engine.py:90-100`):

```
run_base(img):
    img = GeometryProcessor(geometry).process(img, ctx)   # LINEAR float32
    return NormalizationProcessor(process).process(img, ctx)  # -> log space
```

After this change:

```
run_base(img):
    img = GeometryProcessor(geometry).process(img, ctx)   # LINEAR float32
    img = apply_flatfield(img, flatfield_map)              # still LINEAR float32
    return NormalizationProcessor(process).process(img, ctx)  # -> log space
```

The flat-field map is loaded once, resized to match the current image geometry, and then used as a divisor. The operation: `corrected = img / flat_field_normalized`.

## Files to Create/Modify

| Action | File | Purpose |
|--------|------|---------|
| CREATE | `negpy/features/flatfield/__init__.py` | Package init |
| CREATE | `negpy/features/flatfield/logic.py` | `load_flatfield()`, `apply_flatfield()` |
| CREATE | `tests/test_flatfield.py` | Unit tests for logic functions |
| MODIFY | `negpy/cli/batch.py` | Add `--flat-field` CLI flag, apply before pipeline |
| MODIFY | `tests/test_cli_batch.py` | Tests for the new flag |
| MODIFY | `docs/CLI.md` | Document `--flat-field` option |

**Not modified (no GUI or engine changes):**
- `negpy/services/rendering/engine.py` — we do NOT touch the shared `DarkroomEngine`. The flat-field division is applied to the raw buffer in the CLI's `main()` before calling `process_export()`. This avoids breaking the GUI or altering the shared pipeline config model.
- `negpy/domain/models.py` — no new config fields needed.
- `negpy/services/rendering/gpu_engine.py` — no GPU shader changes.

## Implementation

### 1. `negpy/features/flatfield/logic.py`

Two pure functions, no class needed:

```python
def load_flatfield(path: str) -> np.ndarray:
    """
    Loads a flat-field reference frame and returns a normalized float32
    illumination map with values centered around 1.0.

    Accepts TIFF, JPEG, or RAW files (same formats NegPy supports).
    Converts to linear float32 [0-1], then normalizes by dividing
    by the per-channel mean so the map represents *relative* illumination
    (1.0 = average brightness, <1.0 = dimmer, >1.0 = brighter).
    """
```

- Use `loader_factory.get_loader(path)` to load just like any scan
- Demosaic with `rawpy.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16)` → `uint16_to_float32()`
- Normalize each channel: `flat /= mean_per_channel` so average is 1.0
- Clamp minimum to epsilon (1e-6) to avoid division by zero
- Return `np.ndarray` float32 shape (H, W, 3)

```python
def apply_flatfield(image: np.ndarray, flatfield: np.ndarray) -> np.ndarray:
    """
    Divides image by flat-field map to correct uneven illumination.

    Both must be float32. If shapes differ, the flat-field is resized
    (bilinear) to match the image. Output is clipped to [0, 1].
    """
```

- Resize flatfield to match `image.shape[:2]` if different (using `cv2.resize` with `INTER_LINEAR`)
- `result = image / flatfield`
- `np.clip(result, 0.0, 1.0)`
- Return float32

**Key reused component:**
- `loader_factory.get_loader()` from `negpy/infrastructure/loaders/factory.py` — same loader used by `ImageProcessor.process_export()` for loading any supported image format

### 2. `negpy/cli/batch.py` changes

Add CLI flag:

```python
parser.add_argument(
    "--flat-field",
    default=None,
    metavar="FILE",
    help="Path to a flat-field reference frame for vignetting correction",
)
```

In `main()`, after parsing args but before the file loop:

```python
flatfield_map = None
if args.flat_field:
    from negpy.features.flatfield.logic import load_flatfield
    flatfield_map = load_flatfield(args.flat_field)
    print(f"Flat-field loaded: {args.flat_field}", file=sys.stderr)
```

The flat-field correction needs to happen **before** `process_export()` is called, because `process_export()` does its own file loading internally. Two approaches:

**Approach: Inject via monkey-patch of the loaded buffer**
This is fragile. Instead:

**Approach: Pre-process the raw buffer and pass it through the pipeline**
`process_export()` loads the file internally and we can't inject a flat-field step into it without modifying `ImageProcessor`. Instead, we replicate the load + correct + pipeline pattern in the CLI for the flat-field case:

1. When `--flat-field` is set, the CLI loads the raw file itself using `loader_factory` + `rawpy.postprocess()` (same way `process_export` does)
2. Applies `apply_flatfield()` to the linear buffer
3. Calls `ImageProcessor.run_pipeline()` instead of `process_export()`, then handles encoding/writing
4. When `--flat-field` is NOT set, the CLI continues using `process_export()` as it does today (no change)

This keeps the shared engine/processor untouched while giving the CLI a flat-field-aware code path.

**Concrete changes in `main()`:**

```python
if flatfield_map is not None:
    # Load raw file ourselves
    f32_buffer = load_raw_to_float32(file_path)
    f32_corrected = apply_flatfield(f32_buffer, flatfield_map)
    # Use original resolution by default, preview size only with --preview flag
    h_orig, w_orig = f32_corrected.shape[:2]
    render_ref = (
        float(APP_CONFIG.preview_render_size)
        if args.preview
        else float(max(h_orig, w_orig))
    )
    # Run pipeline on corrected buffer
    result_buffer, metrics = processor.run_pipeline(
        f32_corrected, config, source_hash,
        render_size_ref=render_ref,
        prefer_gpu=use_gpu,
    )
    # Encode to bytes (TIFF/JPEG) — extracted as helper
    bits = encode_export(result_buffer, config, export_settings)
else:
    # Existing path
    bits, fmt_or_error = processor.process_export(...)
```

We'll extract a small `load_raw_to_float32(path)` helper in `flatfield/logic.py` that mirrors the loading portion of `process_export()`:
- `loader_factory.get_loader(path)` → `raw.postprocess(gamma=(1,1), ...)` → `uint16_to_float32()`

And an `encode_export(buffer, config, export_settings)` helper in `batch.py` that handles the TIFF/JPEG encoding (extracted from the existing export logic in `ImageProcessor.process_export()` lines 173-220).

### 3. `tests/test_flatfield.py`

Tests for `load_flatfield()`:
- **test_uniform_flat_returns_ones**: A uniform-value image should produce a map of all ~1.0
- **test_vignette_pattern_normalizes**: A center-bright image should produce values >1.0 at center, <1.0 at edges, with mean per channel ≈ 1.0
- **test_clamps_minimum**: Flat field with zero pixels should clamp to epsilon, not produce inf/nan

Tests for `apply_flatfield()`:
- **test_identity_correction**: Dividing by all-1.0 map returns the original image unchanged
- **test_shape_mismatch_resizes**: Flat field of different size gets resized to match
- **test_corrects_vignette**: Apply a known vignette pattern, verify output is more uniform than input
- **test_output_clipped**: Result is clipped to [0.0, 1.0], no values above 1.0

### 4. `tests/test_cli_batch.py` additions

- **TestBuildParser**: add `--flat-field` to `test_minimal_args` (assert None), `test_all_flags` (assert value)
- **TestMain**: add `test_flat_field_flag_loads_and_applies` — mock `load_flatfield` and `apply_flatfield`, verify they're called when flag is provided
- **TestMain**: add `test_flat_field_missing_file_returns_error` — non-existent flat-field path → exit code 1

### 5. `docs/CLI.md` update

Add to options table:

```
| `--flat-field FILE` | none | Path to a flat-field reference frame (blank scan) for vignetting / uneven backlight correction |
```

Add usage section:

```markdown
## Flat-Field Correction

If your scanner produces uneven illumination (bright center, dim edges), you can correct it
by providing a flat-field reference frame. This is a scan of a blank/unexposed frame that
captures your scanner's backlight profile.

### How to create a flat-field reference

1. Scan an empty holder or a piece of unexposed film with your usual scanning settings
2. Save the scan as TIFF (recommended) or any format NegPy supports

### Usage

    negpy --flat-field blank_scan.tiff /path/to/scans/

The flat-field frame is loaded once and applied to every file in the batch. The correction
divides each pixel by the normalized reference, compensating for the uneven illumination.
```

## Verification

1. `pytest tests/test_flatfield.py -v` — all flatfield logic tests pass
2. `pytest tests/test_cli_batch.py -v` — all CLI tests pass (old + new)
3. `python -m negpy.cli.batch --help` — shows `--flat-field` in help text
4. Manual: `negpy --flat-field blank.tiff --mode c41 /path/to/scans/` — corrected files should show reduced edge brightness falloff compared to running without `--flat-field`
