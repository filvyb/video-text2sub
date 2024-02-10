import os
import subprocess
import sys
import tempfile
import shlex

import numpy as np
import scipy
import easyocr
import pyass
from PIL import Image


class Frame:
    def __init__(self, image: Image, frame_num: int, ts: pyass.timedelta):
        self.image = image
        self.frame_num = frame_num
        self.ts = ts
        self.hash = None
        self.text = None

    def get_phash(self):
        if self.hash is not None:
            return self.hash

        hash_size = 8
        highfreq_factor = 4

        image = self.image

        img_size = hash_size * highfreq_factor
        image = image.convert('L')
        image = image.resize((img_size, img_size), Image.Resampling.LANCZOS)
        dct = scipy.fft.dct(scipy.fft.dct(image, axis=0), axis=1)
        dctlowfreq = dct[:hash_size, :hash_size]
        med = np.median(dctlowfreq)
        self.hash = dctlowfreq > med
        return self.hash

    def __str__(self):
        return f"Frame {self.frame_num} at {self.ts}s"

    def __repr__(self):
        return f"Frame {self.frame_num} at {self.ts}s"


def _are_similar_frame(f1: Frame, f2: Frame):
    diff = np.count_nonzero(f1.get_phash() != f2.get_phash())
    return diff <= 13


class VideoProcessor:
    def __init__(self, langs: list[str] = None, gpu: bool = False):
        if langs is None:
            langs = ["en"]
        self.lang = langs
        self.frames = []
        self.reader = easyocr.Reader(langs, gpu=gpu)

    def _get_frames(self, videopath: str, rate: int = 1):
        cmd = f"ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate {videopath}"
        cmd = shlex.split(cmd)
        ret = subprocess.run(cmd, capture_output=True, check=True)
        self.framerate = float(ret.stdout.decode("utf-8").strip().split("/")[0])
        if int(self.framerate) < rate:
            raise ValueError("Rate higher than video framerate")
        print(self.framerate)

        with tempfile.TemporaryDirectory() as tmpdir:
            # ret = os.system(f"ffmpeg -i {videopath} -vf fps={rate}/1 {tmpdir}/%d.jpg")
            tmp = os.path.join(tmpdir, "%d.jpg")
            cmd = f"ffmpeg -i {videopath} -vf fps={rate}/1 {tmp}"
            cmd = shlex.split(cmd)
            subprocess.run(cmd, check=True)

            for i, f in enumerate(os.listdir(tmpdir)):
                print(f"Processing frame {i}{f} in {tmpdir}")
                frame_num = int(f.split(".")[0])
                frame_in_sec = self.framerate / (rate + 1)  # TODO: calc timestamp
                ts = frame_in_sec + (frame_num - 1) * self.framerate
                ts = pyass.timedelta(seconds=ts)
                self.frames.append(Frame(Image.open(os.path.join(tmpdir, f)), frame_num, ts))
            self.frames.sort(key=lambda x: x.frame_num)
            print(f"Total frames: {len(self.frames)}")

    def _remove_similar_frames(self):
        i = 0
        while i < len(self.frames) - 1:
            if _are_similar_frame(self.frames[i], self.frames[i + 1]):
                del self.frames[i]
            else:
                i += 1

    def _save_frames(self, path: str):
        os.makedirs(path, exist_ok=True)
        for i, f in enumerate(self.frames):
            f.image.save(os.path.join(path, f"{i}.jpg"))

    def ocr_video(self, videopath: str, rate: int = 1):
        self._get_frames(videopath, rate)
        # self._save_frames("frames")
        self._remove_similar_frames()
        # self._save_frames("frames2")
        for f in self.frames:
            f.text = self.reader.readtext(f.image)
            i = 0

            while i < len(f.text):
                if f.text[i][2] < 0.7:
                    del f.text[i]
                else:
                    i += 1

            print(f.text)

    def make_ass(self, videopath: str = None, rate: int = 1) -> pyass.Script:
        if videopath is not None:
            self.ocr_video(videopath, rate)

        fin_ass = pyass.Script()
        fin_ass.scriptInfo.append(("PlayResX", str(self.frames[0].image.width)))
        fin_ass.scriptInfo.append(("PlayResY", str(self.frames[0].image.height)))
        fin_ass.styles.append(pyass.Style(borderStyle=pyass.BorderStyle.BORDER_STYLE_OPAQUE_BOX))

        for i, x in enumerate(self.frames):
            for y in x.text:
                end_frame = self.frames[i + 1] if i + 1 < len(self.frames) else self.frames[i]
                text = "{\pos(" + str(int(y[0][0][0])) + "," + str(int(y[0][0][0])) + ")} " + y[1]
                fin_ass.events.append(
                    pyass.Event(format=pyass.EventFormat.DIALOGUE, start=self.frames[i].ts, end=end_frame.ts,
                                text=text))

        return fin_ass


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python video-text2sub.py <videopath>")
        print("Use -h for more information")
        exit(1)

    import argparse

    parser = argparse.ArgumentParser(description="Convert video text to .ass subtitle")
    parser.add_argument("videopath", type=str, help="Path to the video you want to OCR")
    parser.add_argument("--lang", "-l", type=str, default="en",
                        help="Select a language supported by EasyOCR you want to OCR")
    parser.add_argument("--rate", "-r", type=int, default=1,
                        help="Amount of frames analyzed per second, must be >= video framerate")
    parser.add_argument("--gpu", action="store_true", default=False, help="Use your GPU for OCR")

    args = parser.parse_args()

    if args.videopath.strip() == "":
        print("File path required")
        exit(1)

    ocrer = VideoProcessor([args.lang.strip()], args.gpu)
    ocrer.ocr_video(args.videopath.strip(), args.rate)
    sub = ocrer.make_ass()
    print(sub)
    with open("sample.ass", "w+", encoding="utf_8_sig") as f:
        pyass.dump(sub, f)
