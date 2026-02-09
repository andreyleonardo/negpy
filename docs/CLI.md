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
| `--no-gpu` | off | Disable GPU acceleration, use CPU only |
| `--settings JSON_FILE` | none | Load base settings from a JSON file |

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

Use `--settings` to load a full set of processing parameters from a JSON file. This is useful for applying the same darkroom settings across batches. The JSON keys match the internal `WorkspaceConfig` flat dictionary format.

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

# CPU-only processing (no GPU)
negpy --no-gpu /path/to/scans/

# Use saved settings from the GUI
negpy --settings my_recipe.json --output ./prints/ /path/to/scans/
```
