import io
import re

import docx
import PyPDF2
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image, ImageEnhance


class PDFTextExtractor:
    def __init__(self, file_content: bytes):
        self.file_content = file_content

    def extract_text(self) -> str:
        """
        Extract text from PDF with adaptive OCR for mixed content
        """
        try:
            with io.BytesIO(self.file_content) as file:
                reader = PyPDF2.PdfReader(file)
                page_texts = []
                total_text_length = 0

                # Gather statistics about each page
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    page_texts.append(page_text)
                    total_text_length += len(page_text.strip())

                # Calculate adaptive threshold based on document characteristics
                avg_text_per_page = total_text_length / max(len(page_texts), 1)
                adaptive_threshold = min(1500, max(500, avg_text_per_page * 0.5))
                print(f"Adaptive threshold: {adaptive_threshold:.1f} chars")

                # Process each page with the adaptive threshold
                all_text = ""
                for page_num, page_text in enumerate(page_texts):
                    if len(page_text.strip()) < adaptive_threshold:
                        print(f"Page {page_num + 1}: Applying OCR")
                        ocr_text = self.extract_images_and_ocr(page_num)

                        if ocr_text and page_text.strip():
                            all_text += self.merge_texts(page_text, ocr_text) + "\n\n"
                        elif ocr_text:
                            all_text += ocr_text + "\n\n"
                        else:
                            all_text += page_text + "\n\n"
                    else:
                        all_text += page_text + "\n\n"

            return all_text.strip()
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            # Fallback to full OCR
            try:
                return self.extract_text_with_ocr()
            except Exception:
                return ""

    def extract_images_and_ocr(self, page_number: int) -> str:
        """
        Extract images from a PDF page and perform OCR
        """
        try:
            images = convert_from_bytes(
                self.file_content,
                first_page=page_number + 1,
                last_page=page_number + 1,
                dpi=300,
            )
            if not images:
                return ""

            img = images[0].convert("L")
            enhancer = ImageEnhance.Contrast(img)
            enhanced_img = enhancer.enhance(1.5)

            ocr_text1 = pytesseract.image_to_string(
                enhanced_img, config="--psm 1 --oem 3"
            )
            ocr_text2 = pytesseract.image_to_string(
                enhanced_img, config="--psm 6 --oem 3"
            )

            if len(ocr_text1) > len(ocr_text2) * 1.2:
                return ocr_text1
            elif len(ocr_text2) > len(ocr_text1) * 1.2:
                return ocr_text2
            else:
                return self.merge_texts(ocr_text1, ocr_text2)
        except Exception as e:
            print(f"OCR error on page {page_number + 1}: {e}")
            return ""

    def merge_texts(self, text1: str, text2: str) -> str:
        """
        Merge two text sources intelligently
        """
        if not text1.strip():
            return text2
        if not text2.strip():
            return text1

        paras1 = [p.strip() for p in text1.split("\n") if p.strip()]
        paras2 = [p.strip() for p in text2.split("\n") if p.strip()]

        result = []

        for para in paras1:
            result.append(para)

        for para in paras2:
            if self.is_likely_garbage(para):
                continue
            if not any(self.is_similar(para, existing) for existing in result):
                result.append(para)

        return "\n\n".join(result)

    def is_similar(self, text1: str, text2: str, threshold: float = 0.6) -> bool:
        """
        Check if two text snippets are similar
        """
        t1 = re.sub(r"[^\w\s]", "", text1.lower())
        t2 = re.sub(r"[^\w\s]", "", text2.lower())
        words1 = set(t1.split())
        words2 = set(t2.split())

        if not words1 or not words2:
            return False

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union >= threshold

    def is_likely_garbage(self, text: str) -> bool:
        """
        Check if text is likely OCR garbage
        """
        if not text or not text.strip():
            return True
        if len(text) > 10 and " " not in text:
            return True
        special_char_ratio = sum(
            1 for c in text if not c.isalnum() and not c.isspace()
        ) / max(len(text), 1)
        if special_char_ratio > 0.3:
            return True
        if any(len(word) > 25 for word in text.split()):
            return True
        return False

    def extract_text_with_ocr(self) -> str:
        """
        Full OCR fallback for entire PDF
        """
        try:
            images = convert_from_bytes(self.file_content, dpi=300)
            text = ""
            for img in images:
                img = img.convert("L")
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.5)
                page_text = pytesseract.image_to_string(img, config="--psm 1 --oem 3")
                text += page_text + "\n\n"
            return text
        except Exception as e:
            print(f"Error during OCR processing: {e}")
            return ""


class DocxTextExtractor:
    def __init__(self, file_content: bytes):
        self.file_content = file_content

    def extract_text(self) -> str:
        return self.extract_text_from_docx()

    def extract_text_from_docx(self) -> str:
        """Extract text from DOCX file"""
        try:
            with io.BytesIO(self.file_content) as file:
                doc = docx.Document(file)
                return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            print(f"Error extracting text from DOCX: {e}")
            return ""


class ImageTextExtractor:
    def __init__(self, file_content: bytes):
        self.file_content = file_content

    def extract_text(self) -> str:
        return self.extract_text_from_image()

    def extract_text_from_image(self) -> str:
        """Extract text from image using OCR"""
        try:
            with io.BytesIO(self.file_content) as file:
                img = Image.open(file)
                img = img.convert("L")  # Convert to grayscale
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.5)
                text = pytesseract.image_to_string(img, config="--psm 3 --oem 3")
                return text
        except Exception as e:
            print(f"Error extracting text from image: {e}")
            return ""


def extract_text_from_file(file_content: bytes, content_type: str) -> str:
    """
    Extract text based on file type
    """
    extractor = None

    if "pdf" in content_type:
        extractor = PDFTextExtractor(file_content)
    elif (
        "docx" in content_type
        or "openxmlformats-officedocument.wordprocessingml.document" in content_type
    ):
        extractor = DocxTextExtractor(file_content)
    elif "image/" in content_type:
        extractor = ImageTextExtractor(file_content)
    else:
        print(f"Unsupported content type: {content_type}")
        return ""

    if extractor:
        return extractor.extract_text()
    return ""
    return ""
