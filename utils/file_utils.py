"""
File handling utilities for working with different file formats
"""
import io
from typing import Optional
import PyPDF2
import docx


def extract_text_from_pdf(file_content: bytes) -> str:
    """
    Extract text from PDF file

    Args:
        file_content: Binary file content

    Returns:
        Extracted text
    """
    try:
        with io.BytesIO(file_content) as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
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
    elif "docx" in content_type or "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in content_type:
        return extract_text_from_docx(file_content)
    elif "text/" in content_type:
        return file_content.decode('utf-8', errors='replace')
    else:
        print(f"Unsupported content type: {content_type}")
        return ""
