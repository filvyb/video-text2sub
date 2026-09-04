# video-text2sub

Convert text visible in video into positioned ASS subtitles using PP-OCRv6.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- [FFmpeg](https://ffmpeg.org/download.html), including `ffprobe`, on `PATH`
- PaddlePaddle 3.2+ for the `paddle_static` engine

## Setup with uv

```bash
uv init --bare --python 3.10
uv add -r requirements.txt
```

## Usage

```bash
uv run python video-text2sub.py input.mp4
```

Add `--gpu` to use a compatible GPU runtime:

```bash
uv run python video-text2sub.py input.mp4 --gpu
```

CPU inference defaults to batch size 1 for both detection and recognition, based on
benchmarking the ONNX Runtime engine. GPU inference defaults to 4 frames per detection
batch and 16 crops per recognition batch. Override either value when tuning for a
different machine:

```bash
uv run python video-text2sub.py input.mp4 --gpu \
  --det-batch-size 8 \
  --rec-batch-size 32
```

`--engine auto` prefers ONNX Runtime on CPU and otherwise uses Paddle static
inference. To install and select ONNX Runtime explicitly:

```bash
uv add -r requirements-onnx.txt
uv run python video-text2sub.py input.mp4 --engine onnxruntime
```

For ONNX GPU inference, replace `onnxruntime` with `onnxruntime-gpu` and add `--gpu`.

### Optional DeepL translation

Set your API key and choose a target language:

```bash
export DEEPL_AUTH_KEY="your-api-key"
uv run python video-text2sub.py input.mp4 --translate-to DE
```

DeepL detects the source language automatically. Use `--translate-from EN` to set it
explicitly or `--keep-original` to include the OCR text above its translation.

## Common options

- `--det-batch-size`, `--rec-batch-size`: override the device-specific defaults;
  larger GPU batches can improve throughput at the cost of VRAM.
- `--det-unclip-ratio`: lower for tighter detection boxes (`1.0`–`1.3` is typical).
- `--track-iou`, `--max-gap`: tune how detections are joined across frames.
- `--samples-per-track`, `--min-rec-score`: tune recognition sampling and confidence.
- `--font-name`, `--font-size-scale`: customize ASS subtitle rendering.
- `--enable-mkldnn`: enable oneDNN/MKLDNN acceleration for Paddle CPU inference.

Run `uv run python video-text2sub.py --help` for the complete list.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
