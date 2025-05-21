# 🍎 Apple Classification, Distance & Count Estimation

[🔗 GitHub Repository](https://github.com/anuragpathak27/Apple-Classification-Distance-and-Count-Estimation/)

This project performs real-time **apple detection**, **classification into 15 categories**, **counting**, and **distance estimation** using computer vision and the **pinhole camera model**. It can be used for smart farming, quality control, or packaging automation — and achieves **less than 5% error in distance estimation**.

---

## 🎯 Key Features

- ✅ Detects apples in images or live video.
- 🔢 Counts total number of apples in the frame.
- 📏 Estimates distance of apples from the camera using **pinhole camera formula**.
- 🍏 Classifies apples into 15 specific types including *Braeburn*, *Golden*, *Granny Smith*, and *rotten* apples.
- ⚙️ Real-time, lightweight, and accurate under controlled conditions.

---

## 🧠 Technologies Used

- **Language**: Python
- **Libraries**:
  - `OpenCV` – image processing
  - `NumPy` – numerical operations
  - `TensorFlow / Keras` or `scikit-learn` – apple classification model
  - `imutils` – frame pre-processing
- **Concepts**:
  - Contour detection
  - HSV filtering & morphological operations
  - CNN-based classification
  - Pinhole camera model for depth estimation

---

## 🍎 Apple Classification Categories

The model classifies apples into the following **15 categories**:

```python
CLASS_NAMES = [
    "apple_6",
    "apple_braeburn_1",
    "apple_crimson_snow_1",
    "apple_golden_1",
    "apple_golden_2",
    "apple_golden_3",
    "apple_granny_smith_1",
    "apple_hit_1",
    "apple_pink_lady_1",
    "apple_red_1",
    "apple_red_2",
    "apple_red_3",
    "apple_red_delicios_1",
    "apple_red_yellow_1",
    "apple_rooten_1"  # rotten apple detection
]
```

- 🍏 Variety Detection: Recognizes specific apple varieties by color and texture.

- 🧪 Spoilage Identification: Detects spoiled or rotten apples (apple_rooten_1) for quality assurance.

## 📐 Distance Estimation Method

Uses the **Pinhole Camera Model**:

Distance = (Real Height of Apple × Focal Length) / Pixel Height in Image


- **Distance** is measured in the same unit as the real height (e.g., cm).
- **Focal Length** should be pre-calibrated using an apple at a known distance.
- This method achieves approximately **±5% error** under good lighting and camera alignment.
