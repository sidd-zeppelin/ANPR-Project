import easyocr

class EasyOCRReader:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])

    def read(self, image):
        return self.reader.readtext(image)
