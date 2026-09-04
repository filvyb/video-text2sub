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

    def test_detection_batches_inputs_and_preserves_result_order(self):
        predict_calls = []

        class FakeDetection:
            def __init__(self, **_kwargs):
                pass

            def predict(self, **kwargs):
                predict_calls.append(kwargs)
                for index, _image in enumerate(kwargs["input"]):
                    yield {
                        "res": {
                            "dt_polys": [
                                np.asarray(
                                    [
                                        [index, 0],
                                        [index + 10, 0],
                                        [index + 10, 5],
                                        [index, 5],
                                    ]
                                )
                            ],
                            "dt_scores": [0.8 + index / 10],
                        }
                    }

        class FakeRecognition:
            def __init__(self, **_kwargs):
                pass

        fake_paddleocr = types.ModuleType("paddleocr")
        fake_paddleocr.TextDetection = FakeDetection
        fake_paddleocr.TextRecognition = FakeRecognition
        config = MODULE.PipelineConfig(engine="paddle_static", det_batch_size=4)
        with patch.dict(sys.modules, {"paddleocr": fake_paddleocr}):
            backend = MODULE.PaddleOCRBackend(config, "cpu")

        detections = backend.detect_batch(
            [np.zeros((10, 20, 3), dtype=np.uint8) for _ in range(3)]
        )

        self.assertEqual(len(predict_calls), 1)
        self.assertEqual(len(predict_calls[0]["input"]), 3)
        self.assertEqual(predict_calls[0]["batch_size"], 3)
        self.assertEqual([batch[0][1] for batch in detections], [0.8, 0.9, 1.0])
        self.assertEqual([batch[0][0][0, 0] for batch in detections], [0, 1, 2])

    def test_detection_rejects_missing_batch_results(self):
        class FakeDetection:
            def __init__(self, **_kwargs):
                pass

            def predict(self, **_kwargs):
                return [{"res": {"dt_polys": [], "dt_scores": []}}]

        class FakeRecognition:
            def __init__(self, **_kwargs):
                pass

        fake_paddleocr = types.ModuleType("paddleocr")
        fake_paddleocr.TextDetection = FakeDetection
        fake_paddleocr.TextRecognition = FakeRecognition
        with patch.dict(sys.modules, {"paddleocr": fake_paddleocr}):
            backend = MODULE.PaddleOCRBackend(
                MODULE.PipelineConfig(engine="paddle_static"), "cpu"
            )

        with self.assertRaisesRegex(RuntimeError, "1 detection results for 2 frames"):
            backend.detect_batch(["first.png", "second.png"])


class VideoProcessorBatchingTests(unittest.TestCase):
    def test_video_frames_are_detected_in_order_in_bounded_batches(self):
        detection_batches = []

        class FakeBackend:
            def detect_batch(self, image_inputs):
                detection_batches.append(list(image_inputs))
                return [[] for _image_input in image_inputs]

            def recognize(self, _samples):
                return []

        config = MODULE.PipelineConfig(rate=2, det_batch_size=3)
        processor = MODULE.VideoProcessor(config=config)

        def get_frames(_videopath, rate, _memory):
            processor.duration = 3.5
            processor.frames = [
                MODULE.Frame(
                    image=Image.new("RGB", (20, 10), (frame_num, 0, 0)),
                    frame_num=frame_num,
                    ts=frame_num / rate,
                )
                for frame_num in range(7)
            ]

        with patch.object(processor, "_get_frames", side_effect=get_frames), patch.object(
            MODULE, "PaddleOCRBackend", return_value=FakeBackend()
        ):
            processor.ocr_video("input.mp4", memory=True)

        self.assertEqual([len(batch) for batch in detection_batches], [3, 3, 1])
        self.assertEqual(
            [int(image[0, 0, 2]) for batch in detection_batches for image in batch],
            list(range(7)),
        )


class DeepLTranslationTests(unittest.TestCase):
    @staticmethod
    def result(text):
        stable_hash = np.zeros((8, 8), dtype=bool)
        track = MODULE.TextTrack.from_observations(
            1,
            [observation(0.0, (10, 20, 110, 40), stable_hash)],
            sample_pool_size=1,
        )
        track.end = 1.0
        return MODULE.TrackResult(
            track=track,
            text=text,
            confidence=0.9,
            agreement=1.0,
            candidates=[],
        )

    def test_translation_deduplicates_text_and_preserves_result_order(self):
        calls = []

        class Response:
            ok = True
            status_code = 200
            reason = "OK"

            def __init__(self, texts):
                self.texts = texts

            def json(self):
                return {
                    "translations": [
                        {"text": f"translated:{text}"} for text in self.texts
                    ]
                }

        def post(url, **kwargs):
            calls.append((url, kwargs))
            return Response(kwargs["json"]["text"])

        processor = MODULE.VideoProcessor()
        processor.results = [
            self.result("Hello"),
            self.result("World"),
            self.result("Hello"),
        ]

        with patch.object(MODULE.requests, "post", side_effect=post), patch.dict(
            MODULE.os.environ, {"DEEPL_AUTH_KEY": "test-key"}
        ):
            translated = processor.translate_with_deepl("DE", "EN")

        self.assertEqual(translated, 3)
        self.assertEqual(calls[0][0], MODULE.DEEPL_API_URL)
        self.assertEqual(
            calls[0][1]["headers"]["Authorization"],
            "DeepL-Auth-Key test-key",
        )
        self.assertEqual(
            calls[0][1]["json"],
            {
                "text": ["Hello", "World"],
                "target_lang": "DE",
                "preserve_formatting": True,
                "source_lang": "EN",
            },
        )
        self.assertEqual(calls[0][1]["timeout"], MODULE.DEEPL_REQUEST_TIMEOUT)
        self.assertEqual(
            [result.text for result in processor.results],
            ["translated:Hello", "translated:World", "translated:Hello"],
        )

    def test_free_key_uses_free_endpoint_and_auto_detects_source(self):
        calls = []

        class Response:
            ok = True
            status_code = 200
            reason = "OK"

            def json(self):
                return {"translations": [{"text": "Bonjour"}]}

        def post(url, **kwargs):
            calls.append((url, kwargs))
            return Response()

        processor = MODULE.VideoProcessor()
        processor.results = [self.result("Hello")]

        with patch.object(MODULE.requests, "post", side_effect=post), patch.dict(
            MODULE.os.environ, {"DEEPL_AUTH_KEY": "test-key:fx"}
        ):
            processor.translate_with_deepl("FR")

        self.assertEqual(calls[0][0], MODULE.DEEPL_API_FREE_URL)
        self.assertNotIn("source_lang", calls[0][1]["json"])
        self.assertEqual(processor.results[0].text, "Bonjour")

    def test_keep_original_places_translation_on_second_line(self):
        class Response:
            ok = True
            status_code = 200
            reason = "OK"

            def json(self):
                return {"translations": [{"text": "Bonjour"}]}

        processor = MODULE.VideoProcessor()
        processor.results = [self.result("Hello")]

        with patch.object(MODULE.requests, "post", return_value=Response()), patch.dict(
            MODULE.os.environ, {"DEEPL_AUTH_KEY": "test-key"}
        ):
            processor.translate_with_deepl("FR", keep_original=True)

        self.assertEqual(processor.results[0].text, "Hello\nBonjour")
        self.assertEqual(
            MODULE.VideoProcessor._escape_ass(processor.results[0].text),
            r"Hello\NBonjour",
        )

    def test_translation_requires_environment_api_key(self):
        processor = MODULE.VideoProcessor()
        processor.results = [self.result("Hello")]

        with patch.dict(MODULE.os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "DEEPL_AUTH_KEY"):
                processor.translate_with_deepl("DE")

    def test_api_error_does_not_replace_recognized_text(self):
        class Response:
            ok = False
            status_code = 456
            reason = "Quota Exceeded"

            def json(self):
                return {"message": "Quota exceeded"}

        processor = MODULE.VideoProcessor()
        processor.results = [self.result("Hello")]

        with patch.object(MODULE.requests, "post", return_value=Response()), patch.dict(
            MODULE.os.environ, {"DEEPL_AUTH_KEY": "test-key"}
        ):
            with self.assertRaisesRegex(RuntimeError, "HTTP 456.*Quota exceeded"):
                processor.translate_with_deepl("DE")

        self.assertEqual(processor.results[0].text, "Hello")

if __name__ == "__main__":
    unittest.main()
