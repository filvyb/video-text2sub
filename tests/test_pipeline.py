import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).parents[1] / "video-text2sub.py"
SPEC = importlib.util.spec_from_file_location("video_text2sub", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def observation(timestamp, box, crop_hash, quality=1.0):
    polygon = np.asarray(
        [
            [box[0], box[1]],
            [box[2], box[1]],
            [box[2], box[3]],
            [box[0], box[3]],
        ],
        dtype=np.float32,
    )
    return MODULE.Observation(
        ts=timestamp,
        polygon=polygon,
        bbox=box,
        det_score=0.9,
        crop_hash=crop_hash,
        crop=Image.new("RGB", (100, 20), "white"),
        quality=quality,
    )


class TrackingTests(unittest.TestCase):
    def test_track_position_is_coordinate_wise_median(self):
        stable_hash = np.zeros((8, 8), dtype=bool)
        track = MODULE.TextTrack.from_observations(
            1,
            [
                observation(0.0, (10, 20, 110, 40), stable_hash),
                observation(0.1, (12, 18, 112, 38), stable_hash),
                observation(0.2, (11, 19, 111, 39), stable_hash),
            ],
            sample_pool_size=5,
        )
        self.assertEqual(track.stable_box(), (11.0, 19.0, 111.0, 39.0))

    def test_persistent_crop_change_splits_track(self):
        config = MODULE.PipelineConfig(
            rate=10,
            crop_change_distance=5,
            change_patience=2,
            min_track_frames=1,
        )
        tracker = MODULE.TextTracker(config, frame_interval=0.1, duration=1.0)
        old_hash = np.zeros((8, 8), dtype=bool)
        new_hash = np.ones((8, 8), dtype=bool)
        box = (10, 20, 110, 40)

        tracker.update([observation(0.0, box, old_hash)])
        tracker.update([observation(0.1, box, old_hash)])
        tracker.update([observation(0.2, box, new_hash)])
        self.assertEqual(len(tracker.finished), 0)
        tracker.update([observation(0.3, box, new_hash)])

        self.assertEqual(len(tracker.finished), 1)
        self.assertAlmostEqual(tracker.finished[0].end, 0.2)
        self.assertAlmostEqual(tracker.active[0].start, 0.2)
        self.assertEqual(tracker.active[0].frame_count, 2)

    def test_short_detector_dropout_keeps_one_track(self):
        config = MODULE.PipelineConfig(rate=10, max_gap=2, min_track_frames=1)
        tracker = MODULE.TextTracker(config, frame_interval=0.1, duration=1.0)
        stable_hash = np.zeros((8, 8), dtype=bool)
        box = (10, 20, 110, 40)

        tracker.update([observation(0.0, box, stable_hash)])
        tracker.update([])
        tracker.update([])
        tracker.update([observation(0.3, box, stable_hash)])
        tracks = tracker.finish_all()

        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].frame_count, 2)


class ConsensusTests(unittest.TestCase):
    def test_default_recognition_cutoff_is_conservative(self):
        self.assertEqual(MODULE.PipelineConfig().min_rec_score, 0.7)

    def candidate(self, text, score):
        sample = MODULE.CropSample(
            ts=0,
            image=Image.new("RGB", (100, 20), "white"),
            det_score=1.0,
            quality=1.0,
        )
        return MODULE.RecognitionCandidate(text=text, score=score, sample=sample)

    def test_similar_repeated_readings_beat_one_confident_error(self):
        text, confidence, agreement = MODULE._consensus(
            [
                self.candidate("Hello world", 0.80),
                self.candidate("Hello wor1d", 0.75),
                self.candidate("unrelated", 0.95),
            ],
            similarity_threshold=0.72,
        )
        self.assertIn(text, {"Hello world", "Hello wor1d"})
        self.assertGreater(agreement, 0.5)
        self.assertGreater(confidence, 0.4)

    def test_normalization_groups_case_and_spacing(self):
        self.assertEqual(MODULE._normalize_text("  Café  TEST "), "café test")
        self.assertEqual(MODULE._edit_similarity("ABC", "abc"), 1.0)


class SubtitleRenderingTests(unittest.TestCase):
    def test_font_size_uses_box_height_and_clamps_extremes(self):
        config = MODULE.PipelineConfig()

        self.assertEqual(MODULE._ass_font_size((0, 0, 100, 53), config), 64)
        self.assertEqual(MODULE._ass_font_size((0, 0, 100, 2), config), 8)
        self.assertEqual(MODULE._ass_font_size((0, 0, 100, 300), config), 240)

    def test_width_fit_uses_spacing_before_horizontal_scaling(self):
        config = MODULE.PipelineConfig(max_char_spacing_ratio=0.15)
        with patch.object(MODULE, "_measure_ass_text", return_value=100.0):
            spacing, horizontal_scale = MODULE._ass_width_fit(
                "test", (0, 0, 112, 20), 40, config
            )

        self.assertEqual(spacing, 4.0)
        self.assertEqual(horizontal_scale, 100)

    def test_width_fit_clamps_spacing_and_uses_bounded_scaling(self):
        config = MODULE.PipelineConfig(
            max_char_spacing_ratio=0.1,
            min_horizontal_scale=75,
            max_horizontal_scale=150,
        )
        with patch.object(MODULE, "_measure_ass_text", return_value=50.0):
            spacing, horizontal_scale = MODULE._ass_width_fit(
                "x", (0, 0, 100, 20), 20, config
            )

        self.assertEqual(spacing, 0.0)
        self.assertEqual(horizontal_scale, 150)

    def test_ass_event_contains_per_track_font_size(self):
        stable_hash = np.zeros((8, 8), dtype=bool)
        track = MODULE.TextTrack.from_observations(
            1,
            [observation(0.0, (10, 20, 110, 73), stable_hash)],
            sample_pool_size=1,
        )
        track.end = 1.0
        processor = MODULE.VideoProcessor()
        processor.size = (1080, 1920)
        processor.results = [
            MODULE.TrackResult(
                track=track,
                text="Example",
                confidence=0.9,
                agreement=1.0,
                candidates=[],
            )
        ]

        with patch.object(MODULE, "_measure_ass_text", return_value=100.0):
            event = processor.make_ass().events[0]

        self.assertEqual(
            event.text,
            r"{\an7\pos(10,20)\fs64\fsp0\fscx100}Example",
        )

    def test_width_fit_can_be_disabled(self):
        stable_hash = np.zeros((8, 8), dtype=bool)
        track = MODULE.TextTrack.from_observations(
            1,
            [observation(0.0, (10, 20, 110, 73), stable_hash)],
            sample_pool_size=1,
        )
        track.end = 1.0
        processor = MODULE.VideoProcessor(
            config=MODULE.PipelineConfig(fit_width=False)
        )
        processor.size = (1080, 1920)
        processor.results = [
            MODULE.TrackResult(
                track=track,
                text="Example",
                confidence=0.9,
                agreement=1.0,
                candidates=[],
            )
        ]

        with patch.object(
            MODULE, "_measure_ass_text", side_effect=AssertionError("must not measure")
        ):
            event = processor.make_ass().events[0]

        self.assertEqual(event.text, r"{\an7\pos(10,20)\fs64}Example")


class PaddleBackendTests(unittest.TestCase):
    def test_auto_engine_prefers_onnx_only_for_cpu(self):
        resolve = MODULE.PaddleOCRBackend._resolve_engine
        with patch.object(MODULE.importlib.util, "find_spec", return_value=object()):
            self.assertEqual(resolve("auto", "cpu"), "onnxruntime")
            self.assertEqual(resolve("auto", "gpu"), "paddle_static")
        with patch.object(MODULE.importlib.util, "find_spec", return_value=None):
            self.assertEqual(resolve("auto", "cpu"), "paddle_static")

    def test_mkldnn_is_disabled_for_detection_and_recognition(self):
        calls = []

        class FakeModel:
            def __init__(self, **kwargs):
                calls.append(kwargs)

        fake_paddleocr = types.ModuleType("paddleocr")
        fake_paddleocr.TextDetection = FakeModel
        fake_paddleocr.TextRecognition = FakeModel
        with patch.dict(sys.modules, {"paddleocr": fake_paddleocr}):
            MODULE.PaddleOCRBackend(
                MODULE.PipelineConfig(engine="paddle_static"),
                "cpu",
            )

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["engine"], "paddle_static")
        self.assertEqual(calls[1]["engine"], "paddle_static")
        self.assertIs(calls[0]["enable_mkldnn"], False)
        self.assertIs(calls[1]["enable_mkldnn"], False)

    def test_onnx_engine_does_not_receive_mkldnn_option(self):
        calls = []

        class FakeModel:
            def __init__(self, **kwargs):
                calls.append(kwargs)

        fake_paddleocr = types.ModuleType("paddleocr")
        fake_paddleocr.TextDetection = FakeModel
        fake_paddleocr.TextRecognition = FakeModel
        with patch.dict(sys.modules, {"paddleocr": fake_paddleocr}):
            backend = MODULE.PaddleOCRBackend(
                MODULE.PipelineConfig(engine="onnxruntime"),
                "cpu",
            )

        self.assertEqual(backend.engine, "onnxruntime")
        self.assertTrue(all(call["engine"] == "onnxruntime" for call in calls))
        self.assertTrue(all("enable_mkldnn" not in call for call in calls))

if __name__ == "__main__":
    unittest.main()
