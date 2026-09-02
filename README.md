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

Install PaddlePaddle using its platform-specific instructions, then install the
remaining packages:

```bash
python -m pip install -r requirements.txt
```

The PP-OCRv6 models are downloaded automatically the first time they are used.

## Usage

```bash
python video-text2sub.py input.mp4
```

Use a PaddlePaddle GPU build with:

```bash
python video-text2sub.py input.mp4 --gpu
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

Useful tuning options:

- `--det-unclip-ratio`: lower for tighter boxes; try `1.0` to `1.3`.
- `--engine`: `auto`, `paddle_static`, or `onnxruntime`. Benchmark both engines on
  the target machine; performance varies by CPU, GPU, and runtime build.
- `--track-iou`: lower if a noisy detector keeps breaking one text line into tracks.
- `--max-gap`: number of missed sampled frames tolerated before ending a track.
- `--crop-change-distance`: pHash distance from `0` to `64` used to notice text
  changing at the same location. Lower values split more readily.
- `--change-patience`: consecutive changed crops required before a track is split.
- `--samples-per-track`: sharp crops sent to PP-OCR recognition and consensus.
- `--min-rec-score`: individual recognition cutoff before consensus. The default is
  `0.6`; readings below that are commonly noise in background signage, while raising
  it toward `0.9` favors precision over recall.
- `--consensus-similarity`: normalized text similarity required to vote together.
- `--enable-mkldnn`: opt into oneDNN/MKLDNN CPU acceleration. It is disabled by
  default for Paddle static inference because some Paddle/PaddleX combinations fail
  while converting model attributes in the oneDNN executor. It is not passed to ONNX
  Runtime.

Run `python video-text2sub.py --help` for the complete list.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
