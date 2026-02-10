# CLI Batch Converter

NegPy includes a command-line interface for batch converting film negatives without the GUI. It uses the same processing pipeline as the desktop application.

## Installation

Install NegPy as a Python package:

```bash
pip install .
```

This registers the `negpy` command in your environment.

You can also run it directly as a module:

```bash
python -m negpy.cli.batch [OPTIONS] FILE_OR_DIR ...
```

## Quick Start

```bash
# Convert all negatives in a folder (C41 color, TIFF 16-bit output)
negpy /path/to/scans/

# Convert a single file as black & white
negpy --mode bw scan_001.dng

# Convert to JPEG in sRGB for web sharing
negpy --mode c41 --format jpeg --color-space srgb --output ./web/ /path/to/scans/

# Convert slide film (E-6)
negpy --mode e6 --output ./slides/ /path/to/chromes/

# Apply a film preset
negpy --preset portra-400 /path/to/scans/
```

## Usage

```
negpy [OPTIONS] FILE_OR_DIR [FILE_OR_DIR ...]
```

You can pass individual files, directories, or a mix of both. Directories are scanned recursively for supported image files.

## Options

| Flag | Default | Description |
| :--- | :--- | :--- |
| `FILE_OR_DIR` (positional) | *required* | One or more input files or directories |
| `--mode` | `c41` | Film type: `c41` (color negative), `bw` (black & white), `e6` (slide) |
| `--format` | `tiff` | Output format: `tiff` (16-bit) or `jpeg` |
| `--output DIR` | `./export` | Output directory (created automatically) |
| `--color-space` | `adobe-rgb` | Output color space (see below) |
| `--density FLOAT` | `1.0` | Print density / brightness |
| `--grade FLOAT` | `2.0` | Contrast grade |
| `--sharpen FLOAT` | `0.25` | Sharpening amount |
| `--dpi INT` | `300` | Export DPI |
| `--print-size CM` | `30.0` | Print long-edge size in centimeters |
| `--original-res` | off | Export at original sensor resolution (ignores `--dpi` and `--print-size`) |
| `--filename-pattern TEMPLATE` | `positive_{{ original_name }}` | Jinja2 filename template (see [TEMPLATING.md](TEMPLATING.md)) |
| `--crop-offset INT` | `1` | Autocrop border offset in pixels (-5 to 20). Positive values crop more into the image, negative values leave more border. Matches the "Crop Offset" slider in the GUI. |
| `--flat-field FILE` | none | Path to a flat-field reference frame (blank scan) for vignetting / uneven backlight correction |
| `--no-gpu` | off | Disable GPU acceleration, use CPU only |
| `--settings JSON_FILE` | none | Load base settings from a JSON file |
| `--preset NAME` | none | Load a film preset by name (e.g. `portra-400`) from `~/.negpy/presets/` |
| `--init-config` | off | Generate default config at `~/.negpy/config.json` and exit |
| `--list-presets` | off | List available presets and exit |

## Loading Priority

Settings are merged in this order (each layer overrides the previous):

```
Built-in defaults
  ↓ overridden by
~/.negpy/config.json              # User's personal defaults (auto-loaded)
  ↓ overridden by
--preset portra-400               # Film-specific preset
  ↓ overridden by
--settings custom.json            # Explicit per-batch settings file
  ↓ overridden by
CLI flags (--density 1.5 etc.)    # Command-line flags always win
```

## Config File (`~/.negpy/config.json`)

The CLI automatically loads `~/.negpy/config.json` on every run if it exists. This lets you set personal defaults so you don't have to repeat the same flags every time.

### Generating the config file

```bash
negpy --init-config
```

This creates `~/.negpy/config.json` with documented defaults and an empty `~/.negpy/presets/` directory. It will not overwrite an existing config.

### Config file structure

The config has two sections:

- **`cli`** — defaults for CLI flags (flat-field path, output directory, mode, format, etc.)
- **`processing`** — darkroom processing parameters (same keys as `--settings` JSON)

```json
{
    "cli": {
        "flat_field": "/path/to/my/flat_field.tiff",
        "output": "./export",
        "mode": "c41",
        "format": "tiff",
        "color_space": "adobe-rgb",
        "no_gpu": false,
        "crop_offset": null,
        "filename_pattern": "positive_{{ original_name }}"
    },
    "processing": {
        "density": 1.0,
        "grade": 2.0,
        "wb_cyan": 0.0,
        "wb_magenta": 0.0,
        "wb_yellow": 0.0,
        "sharpen": 0.25,
        "color_separation": 1.0,
        "saturation": 1.0
    }
}
```

Set `flat_field` to your scanner's flat-field reference path so it's always applied:

```json
{
    "cli": {
        "flat_field": "/home/user/scans/flat_field_epson_v700.tiff"
    },
    "processing": {}
}
```

CLI flags always override config values. For example, `--density 2.0` overrides the `density` in your config.

## Presets

Presets are JSON files stored in `~/.negpy/presets/`. They use the same flat-dict format as the GUI's presets and the `--settings` flag. Each preset captures the processing parameters for a specific film stock or look.

### Creating a preset

Create a JSON file in `~/.negpy/presets/` with the settings you want:

```bash
# Example: ~/.negpy/presets/portra-400.json
```

```json
{
    "process_mode": "C41",
    "density": 1.2,
    "grade": 2.5,
    "wb_cyan": -0.05,
    "wb_magenta": 0.02,
    "wb_yellow": 0.0,
    "color_separation": 1.1,
    "sharpen": 0.3
}
```

```bash
# Example: ~/.negpy/presets/tri-x.json
```

```json
{
    "process_mode": "B&W",
    "density": 1.0,
    "grade": 3.0,
    "sharpen": 0.2,
    "selenium_strength": 0.3
}
```

You only need to include the parameters you want to change. Anything not specified falls back to the config file defaults, then to built-in defaults.

### Listing presets

```bash
negpy --list-presets
```

```
Available presets:
  portra-400
  tri-x
  ektar-100
```

### Using a preset

```bash
# Apply Portra 400 settings to a batch
negpy --preset portra-400 /path/to/scans/

# Preset + CLI override (bump contrast higher than the preset)
negpy --preset portra-400 --grade 3.5 /path/to/scans/

# Preset + flat-field
negpy --preset tri-x --flat-field blank.tiff /path/to/bw_scans/
```

## Color Spaces

| CLI value | Color space |
| :--- | :--- |
| `srgb` | sRGB |
| `adobe-rgb` | Adobe RGB (1998) |
| `prophoto` | ProPhoto RGB |
| `wide-gamut` | Wide Gamut RGB |
| `aces` | ACES |
| `p3` | Display P3 (D65) |
| `rec2020` | Rec. 2020 |
| `greyscale` | Greyscale |

## Supported File Types

The CLI accepts all file types supported by NegPy:

- **RAW**: `.3fr`, `.ari`, `.arw`, `.bay`, `.braw`, `.cr2`, `.cr3`, `.crw`, `.dng`, `.erf`, `.fff`, `.gpr`, `.iiq`, `.k25`, `.kdc`, `.mdc`, `.mef`, `.mos`, `.mrw`, `.nef`, `.nrw`, `.orf`, `.pef`, `.raf`, `.raw`, `.rw2`, `.sr2`, `.srf`, `.srw`, `.x3f`, and more
- **TIFF**: `.tif`, `.tiff`
- **JPEG**: `.jpg`, `.jpeg`

Unsupported files are skipped with a warning. When scanning directories, only supported files are picked up.

## Settings File

Use `--settings` to load a full set of processing parameters from a JSON file. This is useful for applying the same darkroom settings across batches. The JSON keys match the internal `WorkspaceConfig` flat dictionary format (see [Settings Reference](#settings-reference) below).

Example `settings.json`:

```json
{
    "process_mode": "C41",
    "density": 1.2,
    "grade": 2.5,
    "sharpen": 0.3,
    "export_dpi": 600,
    "export_print_size": 40.0
}
```

```bash
negpy --settings settings.json /path/to/scans/
```

CLI flags override values from the settings file. For example, `--density 2.0` will override the `density` value in the JSON.

## Filename Templates

The `--filename-pattern` flag accepts a Jinja2 template string. See [TEMPLATING.md](TEMPLATING.md) for available variables and examples.

```bash
# Include date and color space in filename
negpy --filename-pattern "{{ date }}_{{ original_name }}_{{ colorspace }}" /path/to/scans/

# Simple prefix
negpy --filename-pattern "print_{{ original_name }}" /path/to/scans/
```

## Output

Progress is printed to stderr:

```
Processing 12 file(s) -> /home/user/export
  [1/12] DSC0001 ... OK (2.3s)
  [2/12] DSC0002 ... OK (2.1s)
  [3/12] DSC0003 ... FAILED (decode error)
  ...
Done: 11/12 succeeded in 25.4s
```

## Exit Codes

| Code | Meaning |
| :--- | :--- |
| `0` | All files processed successfully |
| `1` | One or more files failed, or no supported files found |

## Flat-Field Correction

If your scanner produces uneven illumination (bright center, dim edges, or light leaking on the sides), you can correct it by providing a flat-field reference frame with `--flat-field`.

### How to create a flat-field reference

1. Scan an empty film holder or a piece of unexposed (clear) film using your usual scanning settings (same resolution, same exposure)
2. Save the scan as TIFF (recommended) or any format NegPy supports

The flat-field frame captures your scanner's backlight profile. NegPy normalizes it and divides each scan by it, compensating for the uneven illumination. The correction happens in linear space before any tone mapping, so it preserves the full dynamic range.

### Usage

```bash
# Correct vignetting for a whole batch
negpy --flat-field blank_scan.tiff /path/to/scans/

# Combine with other flags
negpy --flat-field blank.tiff --mode bw --grade 3.0 --output ./prints/ /path/to/scans/
```

The flat-field frame is loaded once and applied to every file in the batch. If the flat-field and scan have different resolutions, the flat-field is resized to match each scan automatically.

You can set `flat_field` in your config file (`~/.negpy/config.json`) to apply the same flat-field reference automatically on every run without specifying `--flat-field` each time.

## Settings Reference

These are all available parameters that can be used in `--settings` JSON files, preset files (`~/.negpy/presets/`), and the `processing` section of `~/.negpy/config.json`. All keys are optional — only include what you want to change.

### Exposure

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `density` | float | `1.0` | Print density / brightness. Higher values produce a darker (denser) print. |
| `grade` | float | `2.0` | Contrast grade, like paper hardness in the darkroom. Range: ~0.5 (very soft) to ~5.0 (very hard). |
| `use_camera_wb` | bool | `false` | Use the camera's white balance from EXIF data instead of manual WB. |
| `wb_cyan` | float | `0.0` | Cyan/red color balance. Negative shifts toward red, positive toward cyan. Range: -1.0 to 1.0. |
| `wb_magenta` | float | `0.0` | Magenta/green color balance. Negative shifts toward green, positive toward magenta. Range: -1.0 to 1.0. |
| `wb_yellow` | float | `0.0` | Yellow/blue color balance. Negative shifts toward blue, positive toward yellow. Range: -1.0 to 1.0. |
| `toe` | float | `0.0` | Shadow lift amount. Raises the black point for a faded look. |
| `toe_width` | float | `3.0` | Shadow transition width. Controls how far the toe effect extends into midtones. |
| `toe_hardness` | float | `1.0` | Shadow transition hardness. Higher values create a sharper transition. |
| `shoulder` | float | `0.0` | Highlight compression amount. Rolls off bright highlights. |
| `shoulder_width` | float | `3.0` | Highlight transition width. Controls how far the shoulder extends into midtones. |
| `shoulder_hardness` | float | `1.0` | Highlight transition hardness. Higher values create a sharper rolloff. |

### Lab / Scanner

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `color_separation` | float | `1.0` | Color channel separation strength. Values >1.0 increase inter-channel contrast, <1.0 reduce it. |
| `saturation` | float | `1.0` | Saturation multiplier. 1.0 is neutral, 0.0 is fully desaturated. |
| `clahe_strength` | float | `0.0` | Adaptive local contrast (CLAHE). 0.0 is off, higher values increase local contrast enhancement. |
| `sharpen` | float | `0.25` | Output sharpening amount. 0.0 is no sharpening. |

### Toning

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `paper_profile` | string | `"None"` | Paper simulation profile. Values: `"None"`, `"Neutral RC"`, `"Cool Glossy"`, `"Warm Fiber"`. |
| `selenium_strength` | float | `0.0` | Selenium toning strength. Adds a cool purple-brown tone primarily in shadows. |
| `sepia_strength` | float | `0.0` | Sepia toning strength. Adds warm brown tone primarily in midtones and highlights. |

### Process

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `process_mode` | string | `"C41"` | Film type. Values: `"C41"` (color negative), `"B&W"` (black & white), `"E-6"` (slide/chrome). |
| `e6_normalize` | bool | `true` | Normalize E-6 slides per-channel. Helps with color casts on slide film. |

### Retouch

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dust_remove` | bool | `false` | Enable automatic dust spot removal. |
| `dust_threshold` | float | `0.66` | Dust detection sensitivity. Lower values detect more spots (may include false positives). |
| `dust_size` | int | `4` | Maximum dust spot size in pixels. Spots larger than this are ignored. |

### Geometry

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `autocrop_offset` | int | `2` | Autocrop border offset in pixels (-5 to 20). Positive crops more into the image. |
| `autocrop_ratio` | string | `"3:2"` | Autocrop aspect ratio. Common values: `"3:2"`, `"4:5"`, `"1:1"`, `"16:9"`. |

### Export

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `export_fmt` | string | `"JPEG"` | Output format. Values: `"JPEG"`, `"TIFF"`. |
| `export_color_space` | string | `"Adobe RGB"` | Output color space. Values: `"sRGB"`, `"Adobe RGB"`, `"ProPhoto RGB"`, `"Wide Gamut RGB"`, `"ACES"`, `"Display P3 (D65)"`, `"Rec. 2020"`, `"Greyscale"`. |
| `export_dpi` | int | `300` | Export DPI resolution. |
| `export_print_size` | float | `30.0` | Print long-edge size in centimeters. |
| `use_original_res` | bool | `false` | Export at original sensor resolution (ignores DPI and print size). |
| `filename_pattern` | string | `"positive_{{ original_name }}"` | Jinja2 filename template. See [TEMPLATING.md](TEMPLATING.md). |
| `export_add_border` | bool | `false` | Add a border around the exported image. |
| `export_border_size` | float | `0.0` | Border size (relative to image). |
| `export_border_color` | string | `"#ffffff"` | Border color as hex string. |

### Full settings example

A complete `settings.json` showing all commonly adjusted parameters:

```json
{
    "process_mode": "C41",
    "density": 1.2,
    "grade": 2.5,
    "wb_cyan": -0.05,
    "wb_magenta": 0.02,
    "wb_yellow": 0.0,
    "color_separation": 1.1,
    "saturation": 1.0,
    "sharpen": 0.3,
    "toe": 0.0,
    "shoulder": 0.0,
    "paper_profile": "None",
    "selenium_strength": 0.0,
    "sepia_strength": 0.0,
    "dust_remove": false,
    "autocrop_offset": 1,
    "export_dpi": 300,
    "export_print_size": 30.0
}
```

## Examples

```bash
# Basic C41 color negative batch
negpy /path/to/roll/

# Black & white at high contrast with custom density
negpy --mode bw --grade 3.5 --density 1.3 /path/to/bw_scans/

# High-res TIFF export for printing
negpy --dpi 600 --print-size 50.0 --color-space prophoto /path/to/scans/

# Original resolution export (no resampling)
negpy --original-res /path/to/scans/

# JPEG for quick web previews
negpy --format jpeg --color-space srgb --output ./previews/ /path/to/scans/

# Mix individual files and directories
negpy scan_001.dng scan_002.dng /path/to/more_scans/

# Tighter crop to remove film border remnants
negpy --crop-offset 10 /path/to/scans/

# Correct scanner vignetting with a flat-field reference
negpy --flat-field blank_scan.tiff /path/to/scans/

# CPU-only processing (no GPU)
negpy --no-gpu /path/to/scans/

# Use saved settings from the GUI
negpy --settings my_recipe.json --output ./prints/ /path/to/scans/

# Apply a film preset
negpy --preset portra-400 /path/to/scans/

# Preset with CLI overrides
negpy --preset tri-x --grade 4.0 --density 1.5 /path/to/bw_scans/

# Initialize config file and presets directory
negpy --init-config

# List all available presets
negpy --list-presets
```
