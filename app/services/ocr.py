import pytesseract
from PIL import Image

def ocr_image(image_path: str, lang: str = "eng+fra") -> tuple[str, float]:
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang=lang)

    data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
    confs = []
    for c in data.get("conf", []):
        try:
            v = int(c)
            if v >= 0:
                confs.append(v)
        except:
            pass
    avg = (sum(confs) / len(confs)) if confs else 0.0
    return text, float(avg)
