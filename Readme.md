# Parking Availability Prediction with Ground-Level Imaging

A two-stage system that estimates the probability of available parking spots in a lot using ground-level photos rather than overhead cameras. Built as a final project for **CSCI-631 Foundations of Computer Vision** at Rochester Institute of Technology.

**Authors:** Anushree Das · Justin Sostre

---

## Motivation

Most parking occupancy systems assume access to overhead cameras with a bird's-eye view of the entire lot — a setup that requires administrative approval, installation infrastructure, and ongoing maintenance costs. This project proposes an alternative: use a photo taken at ground level, detect how many cars are visible, and feed that count into a statistical model that incorporates time of day, weather, and lot capacity to estimate the probability that parking spots remain.

The core insight is that while a ground-level photo cannot show every spot, the number of visible cars combined with contextual factors is enough to produce a useful probabilistic estimate of availability.

---

## System overview

```
Ground-level photo
        ↓
  Faster R-CNN          ← pretrained on COCO, threshold = 0.40
        ↓
  Cars detected (r)
        ↓
  Predictor Model  ← inputs: r, time of day, weather coefficient, lot capacity (l)
        ↓
  P(parking spots available)
```

The system has two components:

**1. Car detector** — a pretrained Faster R-CNN (`faster_rcnn_resnet152_v1_640x640`) from the TensorFlow Object Detection API, evaluated on COCO annotations and fine-tuned for ground-level parking lot images. The detection confidence threshold is set to 0.40, which minimises mean error rate on the RIT parking lot dataset.

**2. Probabilistic predictor** — a mathematical model that combines the car count with contextual variables to estimate parking availability. The model is defined as:

```
c(t) = (1 / 2√2π) · exp(-½ · ((t - 12) / 2)²)     # Gaussian car distribution over the day

p(r, l, w) = l / (r · w · c(t))                      # raw availability ratio (returns 1001 if r ≤ 3)

P(p) = p / (p + α)                                    # bounded probability [0, 1], α = 4
```

where `r` is detected cars, `l` is lot capacity, `w` is the weather coefficient, and `t` is the hour of day. The Gaussian `c(t)` peaks at noon (hour 12), reflecting that RIT class schedules concentrate traffic between 11am–1pm.

**Weather coefficients:**

| Weather | Coefficient |
|---|---|
| Sunny | 1.10 |
| Any other | 1.00 |
| Storm | 0.90 |
| Snow | 0.80 |
| Rain | 0.95 |

---

## Results

**Object detection performance** (IoU range 0.50–0.95, pretrained Faster R-CNN on COCO):

| | Small Objects | Medium Objects | Large Objects |
|---|---|---|---|
| Mean Average Precision | 0.119 | 0.446 | 0.606 |
| Mean Average Recall | 0.228 | 0.567 | 0.704 |

The low mAP on small objects is expected — cars far from the camera occupy fewer pixels and are harder to localise precisely. On the RIT parking lot dataset specifically, mAP at threshold 0.40 is **66.56%**, with the model successfully detecting most visible cars in a scene.

**Known limitation:** ground-level perspective causes partially occluded cars (visible to humans but blocked by the car in front) to be missed by the detector. This systematically undercounts cars in dense lots and is the primary source of prediction error.

---

## Input format

The predictor reads from a CSV file with the following columns in order:

```
image_name, num_cars_detected, time_of_picture, weather_coefficient, lot_capacity
```

Example:
```
lot_A_1200.jpg, 14, 12.0, 1.1, 120
lot_B_0900.jpg, 6, 9.0, 0.95, 80
```

---

## Requirements

```
Python 3.7+
TensorFlow 2.x
TensorFlow Object Detection API
opencv-python
numpy
pandas
```

---

## Data

- **Training/evaluation:** [COCO dataset](https://cocodataset.org/) — filtered to the `car` category only
- **Domain evaluation:** Ground-level photos of RIT parking lots, annotated manually, supplemented with parking lot images from Google Images (on-campus data collection was limited by COVID-19 pandemic restrictions during Fall 2020)

Images must capture a broad view of the lot from slightly above eye level so that multiple rows of cars are visible. Photos of only one or two cars in the foreground are not suitable inputs — the model cannot estimate lot occupancy from partial views.

---

## Limitations and future work

**Occlusion.** Ground-level photography means cars behind other cars are frequently missed. A custom fine-tuned dataset training the model specifically on partially occluded vehicles would improve recall in dense lots.

**Alpha tuning.** The `α = 4` value in the probability bounding function was set manually. With sufficient ground-truth data (photos paired with actual occupancy counts), `α` could be learned or made a function of time and detector confidence.

**Predictor model.** The current statistical model is a mathematical approximation. With more data, replacing it with a neural network taking all inputs simultaneously would allow the model to learn non-linear interactions between time, weather, and car count.

**Image quality filtering.** The system currently accepts any image, including close-up shots of one or two cars that provide no useful lot-level information. A binary classifier to reject bad input images (too close, too narrow a field of view) would improve reliability.

---

## Academic context

Final project for **CSCI-631 Foundations of Computer Vision**, Rochester Institute of Technology, Fall 2020. Published as an academic paper: *Predicting Parking Availability with Ground-Level Imaging*, Anushree Das & Justin Sostre.

Demonstrates: object detection with Faster R-CNN, IoU-based evaluation, TensorFlow Object Detection API, probabilistic modelling, and Gaussian distribution modelling of real-world temporal patterns.

---

## Author

**Anushree Das**
[LinkedIn](https://linkedin.com/in/anushree-s-das) · [GitHub](https://github.com/anushreedas) · [Medium](https://anushree-das.medium.com)
