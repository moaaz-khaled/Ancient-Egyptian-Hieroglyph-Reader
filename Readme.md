# Hieroglyphic Text Recognition

An end-to-end pipeline for recognizing and interpreting Ancient Egyptian hieroglyphic texts using Computer Vision and Natural Language Processing.

This project segments hieroglyphic images, classifies individual glyphs, reconstructs the correct symbol order, and converts them into readable transliterations.

---

## Project Overview

Ancient Egyptian hieroglyphs are complex symbols arranged in structured layouts rather than simple left-to-right text.

This project builds an automated system that can:

1. Detect and segment hieroglyphic symbols from an image.
2. Classify each glyph using deep learning.
3. Reconstruct the correct reading order.
4. Convert the glyph sequence into textual representation.

---

## Pipeline

The system consists of the following stages:

### 1. Image Segmentation (Computer Vision)

Segment the hieroglyphic image into candidate glyph regions using segmentation models.

Tools:
- SAM (Segment Anything Model)
- Image processing techniques

Output:
- Bounding boxes or masks for individual glyphs

---

### 2. Glyph Classification (Deep Learning)

Each segmented glyph is classified into its corresponding hieroglyphic class.

Models used:
- ConvNeXt
- CNN-based classifiers

Output:
- Predicted glyph label for each symbol

---

### 3. Symbol Filtering and Cleanup

Remove:
- Noise segments
- Duplicate segments
- Overlapping glyph detections

Output:
- Clean glyph list

---

### 4. Symbol Ordering

Hieroglyphs follow specific reading patterns (left-to-right, right-to-left, top-to-bottom blocks).

This step reconstructs the correct reading order.

---

### 5. Transliteration (NLP)

Convert the glyph sequence into readable transliteration.

Example:

Hieroglyph sequence → Gardiner codes → Latin transliteration

---

## Dataset

The dataset consists of labeled hieroglyphic symbols collected from:

- Ancient Egyptian inscriptions
- Hieroglyphic dictionaries
- Public hieroglyph datasets

Each sample contains:

- Image
- Glyph label
- Class ID

---

## Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- SAM (Segment Anything)
- Deep Learning (CNN / ConvNeXt)

---

## Project Structure


project/
│
├── dataset/
│
├── segmentation/
│
├── classification/
│
├── ordering/
│
├── transliteration/
│
├── models/
│
├── notebooks/
│
└── README.md


---

## Future Improvements

- Improve segmentation accuracy
- Train larger classification models
- Build a full hieroglyphic OCR system
- Add automatic translation to English

---

## Authors

Graduation Project – Computer Science