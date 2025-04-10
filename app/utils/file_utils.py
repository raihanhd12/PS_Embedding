"""
File handling utilities for working with different file formats
"""

import io

import docx
import PyPDF2
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image


def extract_text_from_pdf(file_content: bytes) -> str:
    """
    Extract text from PDF file, including OCR for image-based PDFs.

    Args:
        file_content: Binary file content

    Returns:
        Extracted text
    """
    try:
        with io.BytesIO(file_content) as file:
            reader = PyPDF2.PdfReader(file)
            text = ""

            # First try to extract text directly
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            # If little or no text was extracted, try OCR
            if (
                len(text.strip()) < 100
            ):  # Arbitrary threshold to determine if PDF is mostly images
                text = extract_text_with_ocr(file_content)

            return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_with_ocr(pdf_content: bytes) -> str:
    """
    Use OCR to extract text from PDF with images.

    Args:
        pdf_content: Binary PDF content

    Returns:
        Extracted text from images
    """
    try:
        # Convert PDF to images
        images = convert_from_bytes(pdf_content)

        # Perform OCR on each image
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img) + "\n\n"

        return text
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        return ""


def extract_text_from_docx(file_content: bytes) -> str:
    """
    Extract text from DOCX file

    Args:
        file_content: Binary file content

    Returns:
        Extracted text
    """
    try:
        with io.BytesIO(file_content) as file:
            doc = docx.Document(file)
            return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""


def extract_text_from_image(image_content: bytes) -> str:
    """
    Extract text from image using OCR

    Args:
        image_content: Binary image content

    Returns:
        Extracted text
    """
    try:
        with io.BytesIO(image_content) as file:
            img = Image.open(file)
            text = pytesseract.image_to_string(img)
            return text
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""


def extract_text_from_file(file_content: bytes, content_type: str) -> str:
    """
    Extract text based on file type

    Args:
        file_content: Binary file content
        content_type: MIME content type

    Returns:
        Extracted text
    """
    if "pdf" in content_type:
        return extract_text_from_pdf(file_content)
    elif (
        "docx" in content_type
        or "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        in content_type
    ):
        return extract_text_from_docx(file_content)
    elif "text/" in content_type:
        return file_content.decode("utf-8", errors="replace")
    elif "image/" in content_type:
        return extract_text_from_image(file_content)
    else:
        print(f"Unsupported content type: {content_type}")
        return ""
