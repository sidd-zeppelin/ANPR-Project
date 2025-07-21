from ultralytics import YOLO

class YOLODetector:
    def __init__(self, config):
        self.model = YOLO(config.MODEL_PATH)
        self.plate_class_id = config.PLATE_CLASS_ID
        self.truck_class_id = config.TRUCK_CLASS_ID

    def detect(self, image):
        return self.model.predict(image, verbose=False)[0]
