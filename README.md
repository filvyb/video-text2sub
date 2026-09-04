# video-text2sub

Convert text visible in a video into positioned ASS subtitles.

The OCR pipeline uses PP-OCRv6 detection and recognition, associates detections
across frames, stabilizes their positions, and chooses text using confidence-weighted
consensus over several sharp crops.

## Requirements

- Python 3.11+
- [FFmpeg](https://ffmpeg.org/download.html), including `ffprobe`, on `PATH`
- PaddlePaddle 3.2+ installed for your CPU or CUDA environment when using the
  `paddle_static` engine

```bash
python -m pip install -r requirements.txt
```

## Usage

```bash
python video-text2sub.py input.mp4
```

Use a PaddlePaddle GPU build with:

```bash
python video-text2sub.py input.mp4 --gpu
```

Detection runs on batches of four sampled video frames by default, while recognition
runs on batches of 16 text crops. Increase either batch size to improve GPU throughput
when memory allows:

```bash
python video-text2sub.py input.mp4 --gpu \
  --det-batch-size 8 \
  --rec-batch-size 32
```

The defaults deliberately use a relatively tight PP-OCR detector expansion:

```bash
python video-text2sub.py input.mp4 \
  --det-limit-side-len 1280 \
  --det-unclip-ratio 1.1 \
  --samples-per-track 5
```

The default `--engine auto` uses ONNX Runtime for CPU inference when it is installed,
and otherwise uses Paddle static inference. Install the optional CPU runtime with:

```bash
python -m pip install -r requirements-onnx.txt
python video-text2sub.py input.mp4 --engine onnxruntime
```

For ONNX GPU inference, install `onnxruntime-gpu` instead of `onnxruntime`, then use
`--engine onnxruntime --gpu`. Only one of those ONNX Runtime packages should be
installed in an environment.

### Optional DeepL translation

Put your DeepL API key in the `DEEPL_AUTH_KEY` environment variable and select a
target language:

```bash
export DEEPL_AUTH_KEY="your-api-key"
python video-text2sub.py input.mp4 --translate-to DE
```

The integration calls the DeepL HTTP API directly using the project's `requests`
dependency, and selects the Free or Pro endpoint from the API key. DeepL detects each
source language automatically. To set it explicitly, add (for example)
`--translate-from EN`. Translation replaces the recognized text in the ASS output
while retaining each track's timing and screen position. Identical recognized strings
are sent only once and reused across tracks.

To keep the untranslated OCR text as the first line and place its translation below
it, use:

```bash
python video-text2sub.py input.mp4 --translate-to DE --keep-original
```

Useful tuning options:

- `--det-unclip-ratio`: lower for tighter boxes; try `1.0` to `1.3`.
- `--det-batch-size`: sampled frames submitted together for text detection; defaults
  to `4`. Higher values can improve GPU utilization but use more VRAM.
- `--rec-batch-size`: detected text crops submitted together for recognition;
  defaults to `16`.
- `--engine`: `auto`, `paddle_static`, or `onnxruntime`. Benchmark both engines on
  the target machine; performance varies by CPU, GPU, and runtime build.
- `--translate-to`: opt into DeepL translation with a target language code such as
  `DE`, `FR`, or `EN-US`; requires `DEEPL_AUTH_KEY`.
- `--translate-from`: set the DeepL source language instead of auto-detecting it.
- `--keep-original`: retain the untranslated OCR text above the DeepL translation.
- `--track-iou`: lower if a noisy detector keeps breaking one text line into tracks.
- `--max-gap`: number of missed sampled frames tolerated before ending a track.
- `--crop-change-distance`: pHash distance from `0` to `64` used to notice text
  changing at the same location. Lower values split more readily.
- `--change-patience`: consecutive changed crops required before a track is split.
- `--samples-per-track`: sharp crops sent to PP-OCR recognition and consensus.
- `--min-rec-score`: individual recognition cutoff before consensus. The default is
  `0.7`; readings below `0.6` are almost always noise in background signage, while raising
  it toward `0.9` favors precision over recall.
- `--consensus-similarity`: normalized text similarity required to vote together.
- `--font-size-scale`: ASS font size relative to each track's median detected box
  height. The default `1.2` is calibrated for Arial/libass rendering.
- `--min-font-size` and `--max-font-size`: clamp dynamically calculated font sizes;
  defaults are `8` and `240`.
- `--font-name`: font family used in the ASS file and for measuring text width.
- `--no-fit-width`: disable width matching and emit neither character-spacing nor
  horizontal-scaling overrides. Height-based font sizing remains enabled.
- `--max-char-spacing-ratio`: limits the positive or negative character spacing used
  to fit the median detected box width; the default is `0.15` times the font size.
- `--min-horizontal-scale` and `--max-horizontal-scale`: bound the final horizontal
  scaling used when safe character spacing cannot fit the box; defaults are `75%`
  and `150%`.
- `--enable-mkldnn`: opt into oneDNN/MKLDNN CPU acceleration. It is disabled by
  default for Paddle static inference because some Paddle/PaddleX combinations fail
  while converting model attributes in the oneDNN executor. It is not passed to ONNX
  Runtime.

Run `python video-text2sub.py --help` for the complete list.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
