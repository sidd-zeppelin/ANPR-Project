import os
import cv2
from collections import Counter
from utils.text_utils import levenshtein, clean_ocr_results, match_plate_pattern
from utils.image_utils import warp_perspective
import numpy as np

class PlateProcessor:
    def __init__(self, config, ocr_reader, valid_state_codes, result_file):
        self.config = config
        self.ocr_reader = ocr_reader
        self.valid_state_codes = valid_state_codes
        self.buffer = []
        self.last_logged = None
        self.output_file = result_file

    def process_plate(self, plate_crop, frame_idx):
        warped = warp_perspective(plate_crop)
        ocr_results = self.ocr_reader.read(warped)
        debug_img = warped.copy()

        for bbox, text, conf in ocr_results:
            bbox = np.array(bbox).astype(int)
            cv2.polylines(debug_img, [bbox], True, (0, 255, 0), 2)
            cv2.putText(debug_img, text, tuple(bbox[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        if ocr_results:
            texts = clean_ocr_results(ocr_results)
            combined = "".join(texts)
            self.buffer.append(combined)
            if len(self.buffer) > self.config.BUFFER_SIZE:
                self.buffer.pop(0)

            most_common, freq = Counter(self.buffer).most_common(1)[0]

            if 8 <= len(most_common) <= 10:
                match = match_plate_pattern(most_common)
                if match:
                    matched_text = match.group()
                    if matched_text[:2] in self.valid_state_codes:
                        if not self.last_logged or levenshtein(matched_text, self.last_logged) > 1:
                            print(f"Frame {frame_idx}: {matched_text}")
                            self.output_file.write(f"Frame {frame_idx}: {matched_text}\n")
                            self.output_file.write("-" * 30 + "\n")
                            self.last_logged = matched_text

                            fname = os.path.join(self.config.EXPORTED_PLATES_DIR, f"frame_{frame_idx}_{matched_text}.png")
                            cv2.imwrite(fname, plate_crop)
                        return matched_text, debug_img
                return most_common, debug_img
        return None, debug_img
