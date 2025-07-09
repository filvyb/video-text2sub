# video-text2sub

This is a simple script to OCR the text from a video and convert it to a subtitle file. Inspired by [video-ocr](https://github.com/PinkFloyded/video-ocr/).

## Requirements
- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html) installed and added to PATH
- Install PyTorch and torchvision from the [official website](https://pytorch.org/get-started/locally/) for CUDA support

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python video-text2sub.py input.mp4
```

## Options
```bash
  -h, --help            show this help message and exit
  --lang LANG, -l LANG  Select a language supported by EasyOCR you want to OCR
  --rate RATE, -r RATE  Amount of frames analyzed per second, must be >= video framerate
  --gpu                 Use your GPU for OCR
  --memory, -m          Load all frames into memory
  --output OUTPUT, -o OUTPUT Set output filepath
````


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
