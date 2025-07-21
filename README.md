# License Plate Detection and OCR Pipeline

This project implements a modular pipeline to **detect vehicle license plates in video**, perform **OCR (Optical Character Recognition)** using EasyOCR, and **export** valid cropped plates and their recognized text to an output directory and log file.

---

## 📁 Project Structure

license-plate-project/
├── config/
│ └── config.py
├── detectors/
│ └── yolo_detector.py
├── models/
│ └── license_plate_detector.pt # Your YOLOv8 plate detection model
├── ocr/
│ └── easyocr_reader.py
├── output/
│ ├── exported_plates/ # Cropped plates saved here
│ └── ocr_results.txt # Final OCR outputs
├── processing/
│ └── plate_processor.py
├── utils/
│ ├── image_utils.py
│ └── text_utils.py
├── videos/
│ └── number_plate_1.avi, etc # Input video files
├── main.py # Entry point
├── requirements.txt
└── README.md


## 🚀 How It Works

The pipeline goes through **4 major stages**:

### 1. **Detection**
- Uses a YOLOv8 model (`license_plate_detector.pt`) to detect number plates in each frame of the video.
- Implemented in `detectors/yolo_detector.py`.

### 2. **OCR (Text Recognition)**
- Uses [EasyOCR](https://github.com/JaidedAI/EasyOCR) to extract text from detected plates.
- Implemented in `ocr/easyocr_reader.py`.

### 3. **Post-Processing**
- Crops and saves the detected plates.
- Filters results to include only **8–10 alphanumeric characters**.
- Normalizes text to avoid duplicate detections.
- Ensures **only one image per truck** is exported (first valid one).
- Handled in `processing/plate_processor.py`, `utils/text_utils.py`, and `utils/image_utils.py`.

### 4. **Output**
- Cropped plates saved in: `output/exported_plates/`
- OCR results saved to: `output/ocr_results.txt` in the format:



---

## 📦 Installation

1. **Clone the repo**:
```bash
git clone https://github.com/your_username/license-plate-project.git
cd license-plate-project


pip install -r requirements.txt
