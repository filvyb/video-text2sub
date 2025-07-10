#!/usr/bin/env python3

import os
import shutil
import subprocess
import sys
import tempfile
import shlex

import numpy as np
import scipy
import easyocr
import pyass
from PIL import Image
from tqdm import tqdm


class Frame:
    def __init__(self, image: Image.Image | str, frame_num: int, ts: pyass.timedelta):
        if isinstance(image, str):
            self.image = image
        else:
            self.image = image.convert('L')
        self.frame_num = frame_num
        self.ts = ts
        self.hash = None
        self.text = None

    def get_phash(self):
        if self.hash is not None:
            return self.hash

        hash_size = 8
        highfreq_factor = 4

        if isinstance(self.image, str):
            image = Image.open(self.image)
            image = image.convert('L')
        else:
            image = self.image

        img_size = hash_size * highfreq_factor
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
        self.tempdir = None
        self.size = None

    def _get_frames(self, videopath: str, rate: int = 1, memory: bool = False):
        cmd = f"ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate {videopath}"
        cmd = shlex.split(cmd)
        ret = subprocess.run(cmd, capture_output=True, check=True)
        fps_str = ret.stdout.decode("utf-8").strip().split("/")
        if fps_str[1].strip() == "1":
            self.framerate = float(fps_str[0])
        else:
            num, den = map(float, fps_str)
            self.framerate = num / den
        if int(self.framerate) < rate:
            raise ValueError("Rate higher than video framerate")
        print(self.framerate)

        self.tempdir = tempfile.TemporaryDirectory()
        tmpdir = self.tempdir.name
        # ret = os.system(f"ffmpeg -i {videopath} -vf fps={rate}/1 {tmpdir}/%d.jpg")
        tmp = os.path.join(tmpdir, "%d.jpg")
        cmd = f"ffmpeg -i {videopath} -vf fps={rate}/1 {tmp}"
        cmd = shlex.split(cmd)
        subprocess.run(cmd, check=True)

        print(f"Processing frames")
        for i, f in tqdm(enumerate(os.listdir(tmpdir))):
            frame_num = int(f.split(".")[0])
            ts = (frame_num - 1) / rate
            ts = pyass.timedelta(seconds=ts)
            if not memory:
                self.frames.append(Frame(os.path.join(tmpdir, f), frame_num, ts))
            else:
                self.frames.append(Frame(Image.open(os.path.join(tmpdir, f)), frame_num, ts))
        self.frames.sort(key=lambda x: x.frame_num)
        print(f"Total frames: {len(self.frames)}")

    def _remove_similar_frames(self):
        kept = [self.frames[0]]
        for frame in self.frames[1:]:
            if not _are_similar_frame(kept[-1], frame):
                kept.append(frame)
        self.frames = kept
        print(f"Frames after removing similar: {len(self.frames)}")

    def _save_frames(self, path: str):
        os.makedirs(path, exist_ok=True)
        for i, fr in enumerate(self.frames):
            if isinstance(fr.image, str):
                shutil.copy2(fr.image, path)
            else:
                fr.image.save(os.path.join(path, f"{i}.jpg"))

    def ocr_video(self, videopath: str, rate: int = 1, memory: bool = False):
        self._get_frames(videopath, rate, memory)
        # self._save_frames("frames")
        self._remove_similar_frames()
        # self._save_frames("frames2")
        print(f"OCRing frames")
        for fr in tqdm(self.frames):
            fr.text = self.reader.readtext(fr.image)
            i = 0

            while i < len(fr.text):
                if fr.text[i][2] < 0.7:
                    del fr.text[i]
                else:
                    i += 1

            # print(fr.text)

        if isinstance(self.frames[0].image, str):
            from pymage_size import get_image_size

            img_format = get_image_size(self.frames[0].image)
            width, height = img_format.get_dimensions()
            self.size = (width, height)
        else:
            width = self.frames[0].image.width
            height = self.frames[0].image.height
            self.size = (width, height)

        self.tempdir.cleanup()

    def make_ass(self, videopath: str = None, rate: int = 1) -> pyass.Script:
        if videopath is not None:
            self.ocr_video(videopath, rate)

        fin_ass = pyass.Script()

        width = self.size[0]
        height = self.size[1]

        fin_ass.scriptInfo.append(("PlayResX", str(width)))
        fin_ass.scriptInfo.append(("PlayResY", str(height)))
        fin_ass.styles.append(
            pyass.Style(borderStyle=pyass.BorderStyle.BORDER_STYLE_OPAQUE_BOX, alignment=pyass.Alignment.TOP_LEFT))

        for i, x in enumerate(self.frames):
            for y in x.text:
                end_frame = self.frames[i + 1] if i + 1 < len(self.frames) else self.frames[i]
                text = r"{\pos(" + str(int(y[0][0][0])) + "," + str(int(y[0][0][1])) + ")} " + y[1]
                fin_ass.events.append(
                    pyass.Event(format=pyass.EventFormat.DIALOGUE, start=self.frames[i].ts, end=end_frame.ts,
                                text=text))

        return fin_ass

    def dump_ass(self, output_path: str, videopath: str = None, rate: int = 1):
        sub = self.make_ass(videopath, rate)
        with open(output_path, "w+", encoding="utf_8_sig") as f:
            pyass.dump(sub, f)


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
    parser.add_argument("--rate", "-r", type=int, default=7,
                        help="Amount of frames analyzed per second, must be >= video framerate")
    parser.add_argument("--gpu", action="store_true", default=False, help="Use your GPU for OCR")
    parser.add_argument("--memory", "-m", action="store_true", default=False, help="Load all frames into memory")
    parser.add_argument("--output", "-o", type=str, help="Set output filepath")

    args = parser.parse_args()

    if args.videopath.strip() == "":
        print("File path required")
        exit(1)

    ocrer = VideoProcessor([args.lang.strip()], args.gpu)
    ocrer.ocr_video(args.videopath.strip(), args.rate, args.memory)
    sub = ocrer.make_ass()
    # print(sub)
    if args.output is None:
        outputpath = os.path.join(os.getcwd(), os.path.basename(args.videopath.strip()) + "-ocr.ass")
    else:
        outputpath = args.output.strip()
    with open(outputpath, "w+", encoding="utf_8_sig") as f:
        pyass.dump(sub, f)
