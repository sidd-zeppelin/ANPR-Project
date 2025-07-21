import os

class Config:
    MODEL_PATH = os.path.join("models", "license_plate_detector.pt")
    STATE_CODES_FILE = "state_codes.txt"
    VIDEO_DIR = "videos"
    OUTPUT_TEXT_FILE = os.path.join("output", "ocr_results.txt")
    EXPORTED_PLATES_DIR = os.path.join("output", "exported_plates")
    PLATE_CLASS_ID = 0
    TRUCK_CLASS_ID = 1
    BUFFER_SIZE = 10
