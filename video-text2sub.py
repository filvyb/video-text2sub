#!/usr/bin/env python3

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import scipy.fft
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageOps

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable


Box = tuple[float, float, float, float]


@dataclass
class PipelineConfig:
    rate: int = 7
    det_model: str = "PP-OCRv6_medium_det"
    rec_model: str = "PP-OCRv6_medium_rec"
    engine: str = "auto"
    det_limit_side_len: int = 1280
    det_thresh: float = 0.3
    det_box_thresh: float = 0.5
    det_unclip_ratio: float = 1.1
    track_iou: float = 0.25
    max_gap: int = 2
    crop_padding: float = 0.08
    crop_change_distance: int = 10
    change_patience: int = 2
    samples_per_track: int = 5
    min_track_frames: int = 2
    min_rec_score: float = 0.6
    consensus_similarity: float = 0.72
    rec_upscale: float = 2.0
    rec_batch_size: int = 16
    enable_mkldnn: bool = False


@dataclass
class Frame:
    image: Image.Image | str
    frame_num: int
    ts: float

    def open(self) -> Image.Image:
        if isinstance(self.image, str):
            with Image.open(self.image) as image:
                return image.convert("RGB")
        return self.image.convert("RGB").copy()

    def paddle_input(self):
        if isinstance(self.image, str):
            return self.image
        # PaddleOCR's ndarray convention follows OpenCV (BGR).
        return np.asarray(self.image.convert("RGB"))[:, :, ::-1].copy()


@dataclass
class Observation:
    ts: float
    polygon: np.ndarray
    bbox: Box
    det_score: float
    crop_hash: np.ndarray
    crop: Image.Image
    quality: float


@dataclass
class CropSample:
    ts: float
    image: Image.Image
    det_score: float
    quality: float


@dataclass
class RecognitionCandidate:
    text: str
    score: float
    sample: CropSample


@dataclass
class TextTrack:
    track_id: int
    sample_pool_size: int
    boxes: list[Box] = field(default_factory=list)
    samples: list[CropSample] = field(default_factory=list)
    pending: list[Observation] = field(default_factory=list)
    reference_hash: np.ndarray | None = None
    start: float = 0.0
    last_seen: float = 0.0
    missed: int = 0
    end: float | None = None

    @classmethod
    def from_observations(
        cls, track_id: int, observations: Sequence[Observation], sample_pool_size: int
    ) -> "TextTrack":
        track = cls(track_id=track_id, sample_pool_size=sample_pool_size)
        for observation in observations:
            track._append(observation)
        return track

    @property
    def association_box(self) -> Box:
        if self.pending:
            return self.pending[-1].bbox
        return self.boxes[-1]

    @property
    def last_observed(self) -> float:
        if self.pending:
            return self.pending[-1].ts
        return self.last_seen

    @property
    def frame_count(self) -> int:
        return len(self.boxes)

    def _append(self, observation: Observation) -> None:
        if not self.boxes:
            self.start = observation.ts
        self.boxes.append(observation.bbox)
        self.last_seen = observation.ts
        self.reference_hash = observation.crop_hash
        self.missed = 0
        self._consider_sample(observation)

    def _consider_sample(self, observation: Observation) -> None:
        sample = CropSample(
            ts=observation.ts,
            image=observation.crop,
            det_score=observation.det_score,
            quality=observation.quality,
        )
        self.samples.append(sample)
        if len(self.samples) > self.sample_pool_size:
            worst = min(range(len(self.samples)), key=lambda index: self.samples[index].quality)
            del self.samples[worst]

    def add(
        self, observation: Observation, change_distance: int, change_patience: int
    ) -> list[Observation] | None:
        """Add an observation, returning a confirmed new visual segment if found."""
        self.missed = 0
        if self.reference_hash is None:
            self._append(observation)
            return None

        if _hash_distance(self.reference_hash, observation.crop_hash) <= change_distance:
            # A transient blur/change did not persist. Retain its geometry but do not
            # let it replace the best recognition samples.
            for pending in self.pending:
                self.boxes.append(pending.bbox)
                self.last_seen = pending.ts
            self.pending.clear()
            self._append(observation)
            return None

        if self.pending and _hash_distance(
            self.pending[-1].crop_hash, observation.crop_hash
        ) > change_distance:
            self.pending = [observation]
            return None

        self.pending.append(observation)
        if len(self.pending) >= change_patience:
            changed = self.pending
            self.pending = []
            return changed
        return None

    def stable_box(self) -> Box:
        coordinates = np.asarray(self.boxes, dtype=np.float32)
        return tuple(float(value) for value in np.median(coordinates, axis=0))

    def recognition_samples(self, limit: int) -> list[CropSample]:
        selected = sorted(self.samples, key=lambda sample: sample.quality, reverse=True)[:limit]
        return sorted(selected, key=lambda sample: sample.ts)


@dataclass
class TrackResult:
    track: TextTrack
    text: str
    confidence: float
    agreement: float
    candidates: list[RecognitionCandidate]


def _phash(image: Image.Image) -> np.ndarray:
    gray = ImageOps.autocontrast(image.convert("L")).resize(
        (32, 32), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(gray, dtype=np.float32)
    dct = scipy.fft.dct(scipy.fft.dct(pixels, axis=0), axis=1)
    low_frequency = dct[:8, :8]
    # Excluding the DC term makes the hash less sensitive to brightness changes.
    median = np.median(low_frequency.ravel()[1:])
    return low_frequency > median


def _hash_distance(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.count_nonzero(left != right))


def _bbox(polygon: np.ndarray) -> Box:
    return (
        float(np.min(polygon[:, 0])),
        float(np.min(polygon[:, 1])),
        float(np.max(polygon[:, 0])),
        float(np.max(polygon[:, 1])),
    )


def _box_iou(left: Box, right: Box) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0:
        return 0.0
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1.0)


def _box_match_cost(left: Box, right: Box, minimum_iou: float) -> float | None:
    left_width = max(left[2] - left[0], 1.0)
    left_height = max(left[3] - left[1], 1.0)
    right_width = max(right[2] - right[0], 1.0)
    right_height = max(right[3] - right[1], 1.0)
    left_center = ((left[0] + left[2]) / 2, (left[1] + left[3]) / 2)
    right_center = ((right[0] + right[2]) / 2, (right[1] + right[3]) / 2)
    dx = abs(left_center[0] - right_center[0]) / max(left_width, right_width)
    dy = abs(left_center[1] - right_center[1]) / max(left_height, right_height)
    size_ratio = max(
        left_width / right_width,
        right_width / left_width,
        left_height / right_height,
        right_height / left_height,
    )
    iou = _box_iou(left, right)

    # The center fallback keeps a track alive when a detector box expands or
    # contracts enough to make IoU misleading.
    center_match = dx <= 0.45 and dy <= 0.55 and size_ratio <= 2.5
    if iou < minimum_iou and not center_match:
        return None
    return (1.0 - iou) + 0.20 * math.hypot(dx, dy) + 0.05 * (size_ratio - 1.0)


def _crop(image: Image.Image, box: Box, padding: float) -> Image.Image:
    width, height = image.size
    box_height = max(box[3] - box[1], 1.0)
    pad = box_height * padding
    left = max(0, math.floor(box[0] - pad))
    top = max(0, math.floor(box[1] - pad))
    right = min(width, math.ceil(box[2] + pad))
    bottom = min(height, math.ceil(box[3] + pad))
    return image.crop((left, top, max(right, left + 1), max(bottom, top + 1)))


def _sharpness(image: Image.Image) -> float:
    pixels = np.asarray(image.convert("L"), dtype=np.float32)
    if pixels.shape[0] < 3 or pixels.shape[1] < 3:
        return 0.0
    laplacian = (
        -4 * pixels[1:-1, 1:-1]
        + pixels[:-2, 1:-1]
        + pixels[2:, 1:-1]
        + pixels[1:-1, :-2]
        + pixels[1:-1, 2:]
    )
    return float(np.var(laplacian))


def _result_payload(result) -> Mapping:
    if isinstance(result, Mapping):
        payload = result
    else:
        payload = getattr(result, "json", None)
        if callable(payload):
            payload = payload()
    if not isinstance(payload, Mapping):
        raise TypeError("PaddleOCR returned an unsupported result object")
    if isinstance(payload.get("res"), Mapping):
        payload = payload["res"]
    return payload


class PaddleOCRBackend:
    def __init__(self, config: PipelineConfig, device: str):
        try:
            from paddleocr import TextDetection, TextRecognition
        except ImportError as error:
            raise RuntimeError(
                "PaddleOCR is not installed. Install PaddlePaddle for your platform, "
                "then run: pip install -r requirements.txt"
            ) from error

        self.config = config
        self.engine = self._resolve_engine(config.engine, device)
        common_options = {
            "device": device,
            "engine": self.engine,
        }
        if self.engine == "paddle_static":
            common_options["enable_mkldnn"] = config.enable_mkldnn

        self.detector = TextDetection(
            model_name=config.det_model,
            limit_side_len=config.det_limit_side_len,
            limit_type="max",
            thresh=config.det_thresh,
            box_thresh=config.det_box_thresh,
            unclip_ratio=config.det_unclip_ratio,
            **common_options,
        )
        self.recognizer = TextRecognition(
            model_name=config.rec_model,
            **common_options,
        )
        print(f"OCR engine: {self.engine}; detector: {config.det_model}; recognizer: {config.rec_model}")

    @staticmethod
    def _resolve_engine(requested: str, device: str) -> str:
        if requested != "auto":
            return requested
        if device == "cpu" and importlib.util.find_spec("onnxruntime") is not None:
            return "onnxruntime"
        return "paddle_static"

    def detect(self, image_input) -> list[tuple[np.ndarray, float]]:
        results = list(self.detector.predict(image_input, batch_size=1))
        if not results:
            return []
        payload = _result_payload(results[0])
        polygons = payload.get("dt_polys", [])
        scores = payload.get("dt_scores", [1.0] * len(polygons))
        return [
            (np.asarray(polygon, dtype=np.float32), float(score))
            for polygon, score in zip(polygons, scores)
            if len(polygon) >= 4
        ]

    def recognize(self, samples: Sequence[CropSample]) -> list[tuple[str, float]]:
        if not samples:
            return []
        recognized = []
        batch_size = max(1, self.config.rec_batch_size)
        for start in range(0, len(samples), batch_size):
            inputs = []
            for sample in samples[start : start + batch_size]:
                image = sample.image.convert("RGB")
                scale = max(self.config.rec_upscale, 48.0 / max(image.height, 1))
                if scale > 1.0:
                    image = image.resize(
                        (
                            max(1, round(image.width * scale)),
                            max(1, round(image.height * scale)),
                        ),
                        Image.Resampling.LANCZOS,
                    )
                inputs.append(np.asarray(image)[:, :, ::-1].copy())

            output = self.recognizer.predict(input=inputs, batch_size=len(inputs))
            for result in output:
                payload = _result_payload(result)
                recognized.append(
                    (
                        str(payload.get("rec_text", "")).strip(),
                        float(payload.get("rec_score", 0.0)),
                    )
                )
        if len(recognized) != len(samples):
            raise RuntimeError(
                f"PaddleOCR returned {len(recognized)} recognition results for "
                f"{len(samples)} crops"
            )
        return recognized


class TextTracker:
    def __init__(self, config: PipelineConfig, frame_interval: float, duration: float):
        self.config = config
        self.frame_interval = frame_interval
        self.duration = duration
        self.active: list[TextTrack] = []
        self.finished: list[TextTrack] = []
        self.next_id = 1
        self.sample_pool_size = max(config.samples_per_track * 3, config.samples_per_track)

    def _new_track(self, observations: Sequence[Observation]) -> TextTrack:
        track = TextTrack.from_observations(
            self.next_id, observations, sample_pool_size=self.sample_pool_size
        )
        self.next_id += 1
        return track

    def _finish(self, track: TextTrack, end: float | None = None) -> None:
        if end is None:
            end = min(self.duration, track.last_seen + self.frame_interval)
        track.end = max(track.start + self.frame_interval, end)
        self.finished.append(track)

    def update(self, observations: list[Observation]) -> None:
        assignments: list[tuple[int, int]] = []
        if self.active and observations:
            costs = np.full((len(self.active), len(observations)), 1_000_000.0)
            for track_index, track in enumerate(self.active):
                for observation_index, observation in enumerate(observations):
                    cost = _box_match_cost(
                        track.association_box, observation.bbox, self.config.track_iou
                    )
                    if cost is not None:
                        costs[track_index, observation_index] = cost
            rows, columns = linear_sum_assignment(costs)
            assignments = [
                (int(row), int(column))
                for row, column in zip(rows, columns)
                if costs[int(row), int(column)] < 1_000_000.0
            ]

        matched_tracks = {track_index for track_index, _ in assignments}
        matched_observations = {observation_index for _, observation_index in assignments}
        replacements: dict[int, TextTrack] = {}
        for track_index, observation_index in assignments:
            track = self.active[track_index]
            changed = track.add(
                observations[observation_index],
                self.config.crop_change_distance,
                self.config.change_patience,
            )
            if changed:
                self._finish(track, end=changed[0].ts)
                replacements[track_index] = self._new_track(changed)

        survivors: list[TextTrack] = []
        for track_index, track in enumerate(self.active):
            if track_index in replacements:
                survivors.append(replacements[track_index])
                continue
            if track_index in matched_tracks:
                survivors.append(track)
                continue
            track.missed += 1
            if track.missed > self.config.max_gap:
                self._finish(track)
            else:
                survivors.append(track)

        for observation_index, observation in enumerate(observations):
            if observation_index not in matched_observations:
                survivors.append(self._new_track([observation]))
        self.active = survivors

    def finish_all(self) -> list[TextTrack]:
        for track in self.active:
            end = self.duration if self.duration - track.last_seen <= (
                self.config.max_gap + 1
            ) * self.frame_interval else None
            self._finish(track, end=end)
        self.active = []
        return sorted(self.finished, key=lambda track: (track.start, track.track_id))


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.split()).casefold()


def _edit_similarity(left: str, right: str) -> float:
    left = _normalize_text(left)
    right = _normalize_text(right)
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return 1.0 - previous[-1] / max(len(left), len(right))


def _consensus(
    candidates: list[RecognitionCandidate], similarity_threshold: float
) -> tuple[str, float, float]:
    if not candidates:
        return "", 0.0, 0.0

    parents = list(range(len(candidates)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            if _edit_similarity(candidates[left].text, candidates[right].text) >= similarity_threshold:
                union(left, right)

    clusters: dict[int, list[int]] = {}
    for index in range(len(candidates)):
        clusters.setdefault(find(index), []).append(index)

    weights = [max(candidate.score, 0.05) for candidate in candidates]
    winner = max(clusters.values(), key=lambda group: sum(weights[index] for index in group))
    total_weight = sum(weights)
    winner_weight = sum(weights[index] for index in winner)
    agreement = winner_weight / max(total_weight, 1e-6)

    def representative_score(index: int) -> float:
        similarities = [
            _edit_similarity(candidates[index].text, candidates[other].text) for other in winner
        ]
        return weights[index] + sum(similarities) / len(similarities)

    representative = max(winner, key=representative_score)
    mean_confidence = sum(candidates[index].score for index in winner) / len(winner)
    confidence = mean_confidence * agreement
    return candidates[representative].text, confidence, agreement


class VideoProcessor:
    def __init__(
        self,
        langs: list[str] | None = None,
        gpu: bool = False,
        config: PipelineConfig | None = None,
    ):
        self.langs = langs or ["en"]
        self.gpu = gpu
        self.config = config or PipelineConfig()
        self.frames: list[Frame] = []
        self.tracks: list[TextTrack] = []
        self.results: list[TrackResult] = []
        self.tempdir: tempfile.TemporaryDirectory | None = None
        self.size: tuple[int, int] | None = None
        self.duration = 0.0
        self.framerate = 0.0

    def _probe_video(self, videopath: str) -> None:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,duration:format=duration",
            "-of",
            "json",
            videopath,
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
        self.size = (int(stream["width"]), int(stream["height"]))
        numerator, denominator = stream.get("avg_frame_rate", "0/1").split("/")
        self.framerate = float(numerator) / max(float(denominator), 1.0)
        self.duration = 0.0
        for duration in (stream.get("duration"), payload.get("format", {}).get("duration")):
            try:
                self.duration = float(duration)
            except (TypeError, ValueError):
                continue
            if math.isfinite(self.duration) and self.duration > 0:
                break
            self.duration = 0.0

    def _get_frames(self, videopath: str, rate: int, memory: bool) -> None:
        if rate <= 0:
            raise ValueError("Rate must be greater than zero")
        self._probe_video(videopath)
        self.tempdir = tempfile.TemporaryDirectory(prefix="video-text2sub-")
        pattern = os.path.join(self.tempdir.name, "%08d.png")
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            videopath,
            "-vf",
            f"fps={rate}/1",
            pattern,
        ]
        subprocess.run(command, check=True)

        paths = sorted(Path(self.tempdir.name).glob("*.png"))
        for index, path in enumerate(paths):
            image: Image.Image | str
            if memory:
                with Image.open(path) as opened:
                    image = opened.convert("RGB").copy()
            else:
                image = str(path)
            self.frames.append(Frame(image=image, frame_num=index, ts=index / rate))
        if not self.frames:
            raise RuntimeError("FFmpeg did not produce any frames")
        if self.duration <= 0:
            self.duration = self.frames[-1].ts + 1.0 / rate

    def _observations(
        self,
        image: Image.Image,
        timestamp: float,
        detections: Sequence[tuple[np.ndarray, float]],
    ) -> list[Observation]:
        observations = []
        for polygon, score in detections:
            box = _bbox(polygon)
            crop = _crop(image, box, self.config.crop_padding)
            quality = math.log1p(_sharpness(crop)) * max(score, 0.05)
            observations.append(
                Observation(
                    ts=timestamp,
                    polygon=polygon,
                    bbox=box,
                    det_score=score,
                    crop_hash=_phash(crop),
                    crop=crop,
                    quality=quality,
                )
            )
        return observations

    def _recognize_tracks(self, backend: PaddleOCRBackend) -> list[TrackResult]:
        eligible = [
            track for track in self.tracks if track.frame_count >= self.config.min_track_frames
        ]
        track_samples = {
            track.track_id: track.recognition_samples(self.config.samples_per_track)
            for track in eligible
        }
        flat_samples = [
            sample for track in eligible for sample in track_samples[track.track_id]
        ]
        recognized = backend.recognize(flat_samples)

        candidates_by_track: dict[int, list[RecognitionCandidate]] = {
            track.track_id: [] for track in eligible
        }
        offset = 0
        for track in eligible:
            for sample in track_samples[track.track_id]:
                text, rec_score = recognized[offset]
                offset += 1
                if text and rec_score >= self.config.min_rec_score:
                    candidates_by_track[track.track_id].append(
                        RecognitionCandidate(
                            text=text,
                            score=rec_score * sample.det_score,
                            sample=sample,
                        )
                    )

        results = []
        for track in tqdm(eligible, desc="Resolving text tracks"):
            candidates = candidates_by_track[track.track_id]
            text, confidence, agreement = _consensus(
                candidates, self.config.consensus_similarity
            )
            results.append(
                TrackResult(
                    track=track,
                    text=text,
                    confidence=confidence,
                    agreement=agreement,
                    candidates=candidates,
                )
            )
        return self._merge_adjacent_results(results)

    def _merge_adjacent_results(self, results: list[TrackResult]) -> list[TrackResult]:
        merged: list[TrackResult] = []
        max_gap = (self.config.max_gap + 1) / self.config.rate
        for result in sorted(results, key=lambda item: (item.track.start, item.track.track_id)):
            if not result.text:
                continue
            previous = next(
                (
                    candidate
                    for candidate in reversed(merged)
                    if _normalize_text(candidate.text) == _normalize_text(result.text)
                    and result.track.start
                    <= (candidate.track.end or candidate.track.last_seen) + max_gap
                    and _box_match_cost(
                        candidate.track.stable_box(),
                        result.track.stable_box(),
                        self.config.track_iou,
                    )
                    is not None
                ),
                None,
            )
            if previous is None:
                merged.append(result)
                continue
            previous.track.end = max(previous.track.end or 0, result.track.end or 0)
            previous.track.boxes.extend(result.track.boxes)
            previous.confidence = max(previous.confidence, result.confidence)
            previous.agreement = max(previous.agreement, result.agreement)
            previous.candidates.extend(result.candidates)
        return merged

    def ocr_video(self, videopath: str, rate: int | None = None, memory: bool = False) -> None:
        if rate is not None:
            self.config.rate = rate
        self.frames = []
        self.tracks = []
        self.results = []
        try:
            self._get_frames(videopath, self.config.rate, memory)
            backend = PaddleOCRBackend(self.config, "gpu" if self.gpu else "cpu")
            tracker = TextTracker(
                self.config,
                frame_interval=1.0 / self.config.rate,
                duration=self.duration,
            )
            for frame in tqdm(self.frames, desc="Detecting and tracking text"):
                detections = backend.detect(frame.paddle_input())
                tracker.update(self._observations(frame.open(), frame.ts, detections))
            self.tracks = tracker.finish_all()
            self.results = self._recognize_tracks(backend)
        finally:
            if self.tempdir is not None:
                self.tempdir.cleanup()
                self.tempdir = None

    @staticmethod
    def _escape_ass(text: str) -> str:
        return text.replace("{", r"\{").replace("}", r"\}").replace("\n", r"\N")

    def make_ass(self, videopath: str | None = None, rate: int | None = None):
        if videopath is not None:
            self.ocr_video(videopath, rate)
        if self.size is None:
            raise RuntimeError("No video has been processed")

        try:
            import pyass
        except ImportError as error:
            raise RuntimeError("pyass is not installed; run: pip install -r requirements.txt") from error

        script = pyass.Script()
        script.scriptInfo.append(("PlayResX", str(self.size[0])))
        script.scriptInfo.append(("PlayResY", str(self.size[1])))
        script.styles.append(
            pyass.Style(
                borderStyle=pyass.BorderStyle.BORDER_STYLE_OPAQUE_BOX,
                alignment=pyass.Alignment.TOP_LEFT,
            )
        )

        for result in self.results:
            x1, y1, _, _ = result.track.stable_box()
            text = rf"{{\an7\pos({round(x1)},{round(y1)})}}" + self._escape_ass(result.text)
            script.events.append(
                pyass.Event(
                    format=pyass.EventFormat.DIALOGUE,
                    start=pyass.timedelta(seconds=result.track.start),
                    end=pyass.timedelta(seconds=result.track.end),
                    text=text,
                )
            )
        return script

    def dump_ass(self, output_path: str, videopath: str | None = None, rate: int | None = None):
        script = self.make_ass(videopath, rate)
        import pyass

        with open(output_path, "w", encoding="utf_8_sig") as output:
            pyass.dump(script, output)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert video text to positioned ASS subtitles")
    parser.add_argument("videopath", help="Video to OCR")
    parser.add_argument(
        "--lang",
        "-l",
        default="en",
        help="Compatibility hint; choose another recognition model for unsupported languages",
    )
    parser.add_argument("--rate", "-r", type=int, default=7, help="Frames analyzed per second")
    parser.add_argument("--gpu", action="store_true", help="Run PaddleOCR on the GPU")
    parser.add_argument("--memory", "-m", action="store_true", help="Load sampled frames into RAM")
    parser.add_argument("--output", "-o", help="Output ASS path")

    parser.add_argument("--det-model", default="PP-OCRv6_medium_det")
    parser.add_argument("--rec-model", default="PP-OCRv6_medium_rec")
    parser.add_argument(
        "--engine",
        choices=("auto", "paddle_static", "onnxruntime"),
        default="auto",
        help="OCR inference engine; auto prefers ONNX Runtime when installed on CPU",
    )
    parser.add_argument("--det-limit-side-len", type=int, default=1280)
    parser.add_argument("--det-thresh", type=float, default=0.3)
    parser.add_argument("--det-box-thresh", type=float, default=0.5)
    parser.add_argument(
        "--det-unclip-ratio",
        type=float,
        default=1.1,
        help="Text box expansion; lower values produce tighter PP-OCR boxes",
    )
    parser.add_argument("--track-iou", type=float, default=0.25)
    parser.add_argument("--max-gap", type=int, default=2, help="Missed samples tolerated per track")
    parser.add_argument("--crop-padding", type=float, default=0.08)
    parser.add_argument(
        "--crop-change-distance",
        type=int,
        default=10,
        help="0-64 pHash distance that starts a new visual text segment",
    )
    parser.add_argument("--change-patience", type=int, default=2)
    parser.add_argument("--samples-per-track", type=int, default=5)
    parser.add_argument("--min-track-frames", type=int, default=2)
    parser.add_argument(
        "--min-rec-score",
        type=float,
        default=0.6,
        help="Discard individual OCR readings below this confidence before consensus",
    )
    parser.add_argument("--consensus-similarity", type=float, default=0.72)
    parser.add_argument("--rec-upscale", type=float, default=2.0)
    parser.add_argument("--rec-batch-size", type=int, default=16)
    parser.add_argument(
        "--enable-mkldnn",
        action="store_true",
        help="Enable oneDNN/MKLDNN CPU acceleration (off by default for compatibility)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.videopath.strip():
        parser.error("video path is required")

    config = PipelineConfig(
        rate=args.rate,
        det_model=args.det_model,
        rec_model=args.rec_model,
        engine=args.engine,
        det_limit_side_len=args.det_limit_side_len,
        det_thresh=args.det_thresh,
        det_box_thresh=args.det_box_thresh,
        det_unclip_ratio=args.det_unclip_ratio,
        track_iou=args.track_iou,
        max_gap=args.max_gap,
        crop_padding=args.crop_padding,
        crop_change_distance=args.crop_change_distance,
        change_patience=args.change_patience,
        samples_per_track=args.samples_per_track,
        min_track_frames=args.min_track_frames,
        min_rec_score=args.min_rec_score,
        consensus_similarity=args.consensus_similarity,
        rec_upscale=args.rec_upscale,
        rec_batch_size=args.rec_batch_size,
        enable_mkldnn=args.enable_mkldnn,
    )
    processor = VideoProcessor([args.lang.strip()], args.gpu, config)
    processor.ocr_video(args.videopath.strip(), memory=args.memory)
    output_path = args.output or os.path.join(
        os.getcwd(), os.path.basename(args.videopath.strip()) + "-ocr.ass"
    )
    processor.dump_ass(output_path)
    print(f"Wrote {len(processor.results)} text tracks to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
