"""
File handling utilities for working with different file formats
"""

import io
import os
import re
import time

import docx
import PyPDF2
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image, ImageEnhance


def extract_text_from_pdf(file_content: bytes) -> str:
    """
    Extract text from PDF file, with special handling for image-based PDFs.
    This optimized version always tries both direct extraction and OCR,
    then combines the results for maximum text coverage.

    Args:
        file_content: Binary file content

    Returns:
        Extracted text
    """
    try:
        with io.BytesIO(file_content) as file:
            reader = PyPDF2.PdfReader(file)

            # Try direct extraction first
            direct_text = ""
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    direct_text += f"[Page {page_num+1} - Direct]\n{page_text}\n\n"

            print(f"Direct extraction found {len(direct_text.strip())} characters")

            # Always run OCR regardless of direct extraction results
            print("Running OCR on PDF...")
            ocr_text = ""
            try:
                ocr_text = extract_text_with_ocr(file_content)
                print(f"OCR found {len(ocr_text.strip())} characters")
            except Exception as ocr_error:
                print(f"OCR encountered an error: {ocr_error}")

            # Combine texts, prioritizing the longer result for each page
            # For simplicity, we'll just concatenate them for now
            combined_text = direct_text + ocr_text

            # Final check
            if len(combined_text.strip()) < 10:
                print("Warning: Both extraction methods produced minimal text")
                if len(reader.pages) > 0:
                    # Force a different OCR approach with higher quality settings
                    try:
                        print("Attempting enhanced OCR with higher settings...")
                        enhanced_text = extract_text_with_enhanced_ocr(file_content)
                        if enhanced_text:
                            return enhanced_text
                    except Exception as enh_error:
                        print(f"Enhanced OCR failed: {enh_error}")

                raise ValueError("Could not extract meaningful text from PDF")

            return combined_text

    except Exception as e:
        print(f"Error during PDF text extraction: {e}")
        # Last resort OCR attempt
        try:
            return extract_text_with_enhanced_ocr(file_content)
        except Exception as last_error:
            print(f"Final OCR attempt failed: {last_error}")
            raise ValueError(f"PDF text extraction failed: {str(e)}")


def extract_text_with_ocr(pdf_content: bytes, page_indices=None) -> str:
    """
    Standard OCR for PDF images.

    Args:
        pdf_content: Binary PDF content
        page_indices: Optional list of page indices to process (0-based)

    Returns:
        Extracted text from images
    """
    try:
        # Convert PDF to images with safe mode
        images = convert_from_bytes(
            pdf_content,
            dpi=300,  # Higher DPI for better quality
            fmt="jpeg",  # Use JPEG instead of PNG to avoid mode issues
            poppler_path=get_poppler_path(),  # Optional: Set path if needed
        )

        # Filter pages if specific indices provided
        if page_indices is not None and page_indices:
            if max(page_indices) < len(images):
                images = [images[i] for i in page_indices]
            else:
                print(f"Warning: Requested page index exceeds PDF length")

        # Configure OCR
        custom_config = r"--oem 3 --psm 6"

        # Perform OCR on each image
        text = ""
        for i, img in enumerate(images):
            try:
                # Ensure image is in RGB mode for compatibility
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Preprocess image for better text detection
                img = preprocess_image(img)

                # Extract text
                page_text = pytesseract.image_to_string(img, config=custom_config)

                # Check for meaningful content (not just noise)
                if is_meaningful_text(page_text):
                    # Add page number and text
                    page_num = page_indices[i] + 1 if page_indices else i + 1
                    text += f"[Page {page_num} - OCR]\n{page_text}\n\n"
                else:
                    print(f"Page {i+1} OCR did not produce meaningful text")

            except Exception as page_error:
                print(f"Error processing page {i} with OCR: {page_error}")
                # Continue with other pages

        return text
    except Exception as e:
        print(f"Error during standard OCR processing: {e}")
        raise


def extract_text_with_enhanced_ocr(pdf_content: bytes) -> str:
    """
    Enhanced OCR with multiple preprocessing techniques for difficult PDFs.

    Args:
        pdf_content: Binary PDF content

    Returns:
        Extracted text from images
    """
    try:
        # Convert with higher DPI
        images = convert_from_bytes(
            pdf_content,
            dpi=600,  # Very high DPI for maximum quality
            fmt="jpeg",
            poppler_path=get_poppler_path(),  # Optional: Set path if needed
        )

        text = ""
        lang = "eng"  # Can be customized based on document language

        for i, img in enumerate(images):
            try:
                # Try multiple preprocessing techniques and combine results
                results = []

                # 1. Original image (converted to RGB)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                results.append(
                    pytesseract.image_to_string(img, lang=lang, config="--psm 6")
                )

                # 2. Grayscale with contrast enhancement
                gray = img.convert("L")
                enhancer = ImageEnhance.Contrast(gray)
                enhanced = enhancer.enhance(2.0)  # Increase contrast
                results.append(
                    pytesseract.image_to_string(enhanced, lang=lang, config="--psm 6")
                )

                # 3. Binary threshold
                threshold = 150
                binary = gray.point(lambda x: 0 if x < threshold else 255, "1")
                results.append(
                    pytesseract.image_to_string(binary, lang=lang, config="--psm 6")
                )

                # 4. Try different PSM mode for layout analysis
                results.append(
                    pytesseract.image_to_string(img, lang=lang, config="--psm 11")
                )

                # Choose the best result (longest that looks meaningful)
                best_result = max(
                    [r for r in results if is_meaningful_text(r)],
                    key=lambda x: len(x),
                    default="",
                )

                if best_result:
                    text += f"[Page {i+1} - Enhanced OCR]\n{best_result}\n\n"

            except Exception as page_error:
                print(f"Error in enhanced OCR for page {i}: {page_error}")

        return text

    except Exception as e:
        print(f"Error during enhanced OCR: {e}")
        raise


def preprocess_image(img):
    """
    Preprocess image to improve OCR quality.

    Args:
        img: PIL Image

    Returns:
        Processed PIL Image
    """
    try:
        # Convert to grayscale
        gray = img.convert("L")

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(1.5)

        # Optional: Apply sharpening
        # sharpener = ImageEnhance.Sharpness(enhanced)
        # enhanced = sharpener.enhance(1.5)

        return enhanced
    except Exception as e:
        print(f"Image preprocessing error: {e}")
        return img  # Return original if processing fails


def is_meaningful_text(text):
    """
    Check if extracted text appears to be meaningful.

    Args:
        text: Extracted text string

    Returns:
        Boolean indicating if text appears meaningful
    """
    if not text or len(text.strip()) < 5:
        return False

    # Count words that appear to be real words (longer than 1 character)
    words = [w for w in re.findall(r"\b[a-zA-Z][a-zA-Z]+\b", text)]
    if len(words) < 2:
        return False

    # Check for excessive garbage characters
    total_chars = len(text.strip())
    alpha_chars = sum(
        c.isalpha() or c.isspace() or c in ",.;:?!()[]{}\"'" for c in text
    )
    if alpha_chars / total_chars < 0.5:  # Less than 50% recognizable characters
        return False

    return True


def get_poppler_path():
    """
    Get poppler path based on environment.
    Override this with your specific path if needed.
    """
    # Example paths for different platforms - adjust as needed
    if os.name == "nt":  # Windows
        return None  # Set your Windows path here if needed
    elif os.name == "posix":  # Linux/Mac
        # Common locations
        for path in ["/usr/local/bin", "/usr/bin", "/opt/homebrew/bin"]:
            if os.path.exists(os.path.join(path, "pdftoppm")):
                return path
    return None


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

            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Preprocess for better results
            img = preprocess_image(img)

            # Try multiple configurations and take the best result
            results = []

            # Standard config
            results.append(pytesseract.image_to_string(img, config="--oem 3 --psm 6"))

            # Alternative configs for different layouts
            results.append(
                pytesseract.image_to_string(img, config="--oem 3 --psm 3")
            )  # Fully automatic layout
            results.append(
                pytesseract.image_to_string(img, config="--oem 3 --psm 4")
            )  # Single column of text

            # Choose the best result
            best_result = max(
                [r for r in results if is_meaningful_text(r)],
                key=lambda x: len(x),
                default="",
            )

            return (
                best_result if best_result else results[0]
            )  # Fallback to first result

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
    try:
        print(f"Processing file with content type: {content_type}")

        if "pdf" in content_type:
            start_time = time.time()
            text = extract_text_from_pdf(file_content)
            elapsed = time.time() - start_time
            print(f"PDF processing took {elapsed:.2f} seconds")

            # Check if we got meaningful text
            if not text or len(text.strip()) < 10:
                print(f"Warning: Insufficient text extracted from PDF")
                raise ValueError("Insufficient text extracted from file")
            return text

        elif (
            "docx" in content_type
            or "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            in content_type
        ):
            text = extract_text_from_docx(file_content)
            if not text or len(text.strip()) < 10:
                raise ValueError("Insufficient text extracted from DOCX file")
            return text

        elif "text/" in content_type:
            return file_content.decode("utf-8", errors="replace")

        elif "image/" in content_type:
            text = extract_text_from_image(file_content)
            if not text or len(text.strip()) < 10:
                raise ValueError("No text detected in image")
            return text

        else:
            raise ValueError(f"Unsupported content type: {content_type}")

    except Exception as e:
        print(f"Error in extract_text_from_file: {str(e)}")
        raise
