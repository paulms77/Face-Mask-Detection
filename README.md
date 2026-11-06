# Face Mask Detection System

Real-time detection of whether a face is wearing a mask — **correctly**,
**incorrectly**, or **not at all** — from a webcam feed or a video file.

Two stages: a Caffe SSD face detector locates faces, then a MobileNetV2
classifier fine-tuned on an augmented dataset judges each one.

---

## What this fixes

Off-the-shelf mask detectors fail in three recurring ways:

1. They only recognise the colour of mask they were trained on
2. They score a chin mask or a nose-out mask as "wearing a mask" — which is the
   case that actually matters
3. Accuracy collapses under unusual lighting or camera angles

All three come from the same root cause: public mask datasets are small and
visually narrow. The answer here is to **synthesise the missing data** rather
than hunt for more photos.

## Dataset

Starting point, collected from Kaggle:

| Class | Images |
| --- | --- |
| Without mask | 8,718 |
| Incorrectly worn | 4,723 |
| With mask | 4,109 |

### Synthetic mask augmentation

<img width="420" src="https://github.com/user-attachments/assets/83946be6-c1a2-4c4e-adbc-cca0a48594dd">

1. **Detect and extract the face** from a bare-faced photo using
   `face_recognition` to get facial landmarks
2. **Composite a mask onto it**, anchored to those landmarks — once worn
   correctly, once worn incorrectly (below the nose, under the chin)

Generated:

| Class | Added |
| --- | --- |
| Without mask | 6,912 |
| Incorrectly worn | 6,270 |
| Correctly worn | 6,270 |

Mask colours are distributed deliberately rather than uniformly — white (796) <
black (1,952) < blue (remainder) — so the classifier cannot use colour as a
proxy for "mask present". Mask assets are in [`mask/`](mask/).

### Image augmentation

| Transform | Why |
| --- | --- |
| `RandomRotation` | Faces are rarely perfectly upright; generalises over alignment error |
| `RandomHorizontalFlip` | Faces are not symmetric; removes left/right bias |
| `brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2` | Simulates indoor/outdoor, poor lighting, different camera settings |

<img width="420" alt="Augmentation result" src="https://github.com/user-attachments/assets/b9f6a6e0-ceb2-41f2-b350-bdac39d45129">

## Model

**MobileNetV2** pretrained on ImageNet, with a custom classification head via
transfer learning. MobileNetV2 is chosen for inference speed — this has to keep
up with a webcam.

<img width="420" alt="Network architecture" src="https://github.com/user-attachments/assets/10a683fc-8ce9-423e-95b7-03d9211f6ef7">

Face detection uses the Caffe ResNet-10 SSD
(`res10_300x300_ssd_iter_140000_fp16.caffemodel` + `deploy.prototxt`).

### Decision thresholds

**Face confidence ≥ 0.30.** The usual 0.50 was tested and rejected: faces were
regularly missed under adverse lighting or at an angle. Lowering to 0.30
recovers them, and the downstream classifier filters the false positives.

**Mask verdict:**

| Condition | Box |
| --- | --- |
| P(mask) > P(no mask) **and** P(mask) ≥ 0.70 | 🟩 green — mask on |
| P(no mask) > P(mask) **and** P(no mask) ≥ 0.70 | 🟥 red — no mask |

## Setup

```bash
git clone https://github.com/paulms77/Face-Mask-Detection.git
cd Face-Mask-Detection

pip install opencv-python torch torchvision face_recognition \
            scikit-learn matplotlib pillow numpy
```

`best_model.pth` (trained weights) and the Caffe face detector are included, so
inference runs without retraining.

## Usage

### Live webcam

```python
vs = cv2.VideoCapture(0)
```

<img src="https://github.com/user-attachments/assets/ab48cbef-94f9-48ac-8c43-5e67f2af9424">

### Video file

```python
video_path = './input.mp4'
vs = cv2.VideoCapture(video_path)
```

<img src="https://github.com/user-attachments/assets/4db15274-5d70-4dff-8441-48a051941afa">

### Notebooks

| Notebook | Purpose |
| --- | --- |
| [`mask-augmentation.ipynb`](mask-augmentation.ipynb) | Build the synthetic dataset — face extraction and mask compositing |
| [`train.ipynb`](train.ipynb) | Fine-tune MobileNetV2 → `best_model.pth` |
| [`face-mask-detection.ipynb`](face-mask-detection.ipynb) | Run detection on webcam or video |

Run them in that order to reproduce from scratch, or go straight to the third to
use the included weights.

## Results

| | |
| --- | --- |
| Training curves | <img width="380" src="https://github.com/user-attachments/assets/ab248597-6823-46f8-91c5-0cf49dc107ff"> |
| Classification report | <img width="380" src="https://github.com/user-attachments/assets/46ec7974-b0bf-481a-b85e-7dbcbba23b56"> |

## Dependencies

[OpenCV](https://opencv.org/) ·
[Caffe face detector](https://caffe.berkeleyvision.org/) ·
[PyTorch](https://pytorch.org/) ·
[MobileNetV2](https://arxiv.org/abs/1801.04381) ·
[face_recognition](https://github.com/ageitgey/face_recognition) ·
[scikit-learn](https://scikit-learn.org/stable/) ·
[Matplotlib](https://matplotlib.org/) ·
[Pillow](https://pypi.org/project/pillow/) ·
[NumPy](https://numpy.org/)

## License

MIT — see [LICENSE](LICENSE).
