"""
Enhanced file handling utilities with adaptive OCR for PDFs with mixed content
"""

import io
import re
from typing import List

import docx
import PyPDF2
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image, ImageEnhance


def extract_text_from_pdf(file_content: bytes) -> str:
    """
    Extract text from PDF with adaptive OCR for mixed content
    """
    try:
        # First pass: analyze document to set adaptive threshold
        with io.BytesIO(file_content) as file:
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
                # If page has limited text or suspicious patterns, apply OCR
                if len(page_text.strip()) < adaptive_threshold:
                    print(f"Page {page_num+1}: Applying OCR")
                    ocr_text = extract_images_and_ocr(file_content, page_num)

                    if ocr_text and page_text.strip():
                        # Combine both texts
                        all_text += merge_texts(page_text, ocr_text) + "\n\n"
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
            return extract_text_with_ocr(file_content)
        except Exception:
            return ""


def extract_images_and_ocr(pdf_content: bytes, page_number: int) -> str:
    """
    Extract images from a PDF page and perform OCR
    """
    try:
        # Convert page to high-resolution image
        images = convert_from_bytes(
            pdf_content, first_page=page_number + 1, last_page=page_number + 1, dpi=300
        )

        if not images:
            return ""

        # Enhance image for better OCR
        img = images[0].convert("L")  # Convert to grayscale
        enhancer = ImageEnhance.Contrast(img)
        enhanced_img = enhancer.enhance(1.5)  # Increase contrast

        # Try different OCR configurations
        ocr_text1 = pytesseract.image_to_string(
            enhanced_img, config="--psm 1 --oem 3"  # Auto page segmentation
        )

        ocr_text2 = pytesseract.image_to_string(
            enhanced_img, config="--psm 6 --oem 3"  # Assume single block of text
        )

        # Use the better result (usually the longer one)
        if len(ocr_text1) > len(ocr_text2) * 1.2:
            return ocr_text1
        elif len(ocr_text2) > len(ocr_text1) * 1.2:
            return ocr_text2
        else:
            # Combine results
            return merge_texts(ocr_text1, ocr_text2)

    except Exception as e:
        print(f"OCR error on page {page_number+1}: {e}")
        return ""


def merge_texts(text1: str, text2: str) -> str:
    """
    Merge two text sources intelligently
    """
    # If one is empty, return the other
    if not text1.strip():
        return text2
    if not text2.strip():
        return text1

    # Split into paragraphs
    paras1 = [p.strip() for p in text1.split("\n") if p.strip()]
    paras2 = [p.strip() for p in text2.split("\n") if p.strip()]

    # Create merged result
    result = []

    # Add text1 paragraphs
    for para in paras1:
        result.append(para)

    # Add unique text2 paragraphs
    for para in paras2:
        # Skip garbage OCR text
        if is_likely_garbage(para):
            continue

        # Check if similar paragraph already exists
        if not any(is_similar(para, existing) for existing in result):
            result.append(para)

    return "\n\n".join(result)


def is_similar(text1: str, text2: str, threshold: float = 0.6) -> bool:
    """
    Check if two text snippets are similar
    """
    # Normalize texts
    t1 = re.sub(r"[^\w\s]", "", text1.lower())
    t2 = re.sub(r"[^\w\s]", "", text2.lower())

    # Get words
    words1 = set(t1.split())
    words2 = set(t2.split())

    # Calculate similarity
    if not words1 or not words2:
        return False

    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))

    return intersection / union >= threshold


def is_likely_garbage(text: str) -> bool:
    """
    Check if text is likely OCR garbage
    """
    # Skip empty text
    if not text or not text.strip():
        return True

    # Check for lack of spaces
    if len(text) > 10 and " " not in text:
        return True

    # Check for too many special characters
    special_char_ratio = sum(
        1 for c in text if not c.isalnum() and not c.isspace()
    ) / max(len(text), 1)
    if special_char_ratio > 0.3:
        return True

    # Check for suspiciously long words
    words = text.split()
    if any(len(word) > 25 for word in words):
        return True

    return False


def extract_text_with_ocr(pdf_content: bytes) -> str:
    """
    Full OCR fallback for entire PDF
    """
    try:
        # Convert PDF to images
        images = convert_from_bytes(pdf_content, dpi=300)

        # Perform OCR on each image
        text = ""
        for img in images:
            # Convert to grayscale
            img = img.convert("L")

            # Enhance contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)

            # Perform OCR
            page_text = pytesseract.image_to_string(img, config="--psm 1 --oem 3")
            text += page_text + "\n\n"

        return text
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        return ""


def extract_text_from_file(file_content: bytes, content_type: str) -> str:
    """
    Extract text based on file type
    """
    if "pdf" in content_type:
        return extract_text_from_pdf(file_content)
    elif (
        "docx" in content_type
        or "openxmlformats-officedocument.wordprocessingml.document" in content_type
    ):
        return extract_text_from_docx(file_content)
    elif "text/" in content_type:
        return file_content.decode("utf-8", errors="replace")
    elif "image/" in content_type:
        return extract_text_from_image(file_content)
    else:
        print(f"Unsupported content type: {content_type}")
        return ""


def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from DOCX file"""
    try:
        with io.BytesIO(file_content) as file:
            doc = docx.Document(file)
            return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""


def extract_text_from_image(image_content: bytes) -> str:
    """Extract text from image using OCR"""
    try:
        with io.BytesIO(image_content) as file:
            img = Image.open(file)
            img = img.convert("L")  # Convert to grayscale

            # Enhance contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)

            # Apply OCR
            text = pytesseract.image_to_string(img, config="--psm 3 --oem 3")
            return text
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""
