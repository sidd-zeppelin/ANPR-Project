import cv2
import os
from config.config import Config
from detectors.yolo_detector import YOLODetector
from ocr.easyocr_reader import EasyOCRReader
from processing.plate_processor import PlateProcessor

def load_state_codes(filepath):
    with open(filepath, "r") as f:
        return set(line.strip().upper() for line in f if line.strip())

def main():
    config = Config()
    os.makedirs(config.EXPORTED_PLATES_DIR, exist_ok=True)

    valid_state_codes = load_state_codes(config.STATE_CODES_FILE)
    detector = YOLODetector(config)
    ocr = EasyOCRReader()

    video_path = os.path.join(config.VIDEO_DIR, "number_plate_1.avi")
    cap = cv2.VideoCapture(video_path)
    output_file = open(config.OUTPUT_TEXT_FILE, "w", encoding="utf-8")
    processor = PlateProcessor(config, ocr, valid_state_codes, output_file)

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        results = detector.detect(frame)

        for idx, cls in enumerate(results.boxes.cls.cpu().numpy().astype(int)):
            if cls != config.TRUCK_CLASS_ID:
                continue

            x1, y1, x2, y2 = map(int, results.boxes.xyxy.cpu().numpy()[idx])
            truck_crop = frame[y1:y2, x1:x2]

            sub_results = detector.detect(truck_crop)
            for jdx, sub_cls in enumerate(sub_results.boxes.cls.cpu().numpy().astype(int)):
                if sub_cls != config.PLATE_CLASS_ID:
                    continue

                px1, py1, px2, py2 = map(int, sub_results.boxes.xyxy.cpu().numpy()[jdx])
                plate_crop = truck_crop[py1:py2, px1:px2]
                text, debug = processor.process_plate(plate_crop, frame_idx)

                abs_x1, abs_y1 = x1 + px1, y1 + py1
                abs_x2, abs_y2 = x1 + px2, y1 + py2
                cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (0,255,0), 2)
                if text:
                    cv2.putText(frame, f"Plate: {text}", (abs_x1, abs_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                cv2.imshow("Plate Debug", debug)
                break

        cv2.imshow("Detections", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    output_file.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
