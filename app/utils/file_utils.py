"""
File handling utilities for working with different file formats
Balanced optimization for speed and image-based PDF detection
"""

import io
import os

import docx
import PyPDF2
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image, ImageEnhance


def extract_text_from_pdf(file_content: bytes) -> str:
    """
    Extract text from PDF file with balanced approach.
    Optimized to detect image-based PDFs while maintaining reasonable speed.

    Args:
        file_content: Binary file content

    Returns:
        Extracted text
    """
    try:
        with io.BytesIO(file_content) as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)

            # Quick check for image-based PDF - examine resources in first page
            is_likely_image_based = False
            if total_pages > 0:
                first_page = reader.pages[0]
                # Check for XObject or image resources that often indicate image-based content
                if "/XObject" in first_page or "/Image" in str(first_page):
                    is_likely_image_based = True
                    print("PDF appears to be image-based based on resources")

            # If likely image-based, go straight to OCR for first page
            if is_likely_image_based:
                print("Using OCR for likely image-based PDF...")
                # Try just first page first as a sample
                sample_text = extract_text_with_ocr(file_content, page_indices=[0])
                if (
                    len(sample_text.strip()) > 20
                ):  # If we got meaningful text from sample
                    # OCR the rest of the document
                    if total_pages > 1:
                        return sample_text + extract_text_with_ocr(
                            file_content, page_indices=list(range(1, total_pages))
                        )
                    return sample_text

            # If not clearly image-based, try standard approach
            # Try direct extraction first
            direct_text = ""
            empty_pages = []

            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    direct_text += f"{page_text}\n\n"
                else:
                    empty_pages.append(page_num)

            # If we got substantial direct text, return it
            if len(direct_text.strip()) >= 50:
                return direct_text

            # If most/all pages are empty, this is probably an image-based PDF
            if len(empty_pages) > 0:
                # Run OCR on empty pages
                print(
                    f"Running OCR on {len(empty_pages)} pages without extracted text..."
                )
                ocr_text = extract_text_with_ocr(file_content, page_indices=empty_pages)
                full_text = direct_text + ocr_text

                if len(full_text.strip()) < 10:
                    # If still not enough text, try one more approach with different settings
                    print("Trying higher quality OCR as final attempt...")
                    return extract_text_with_better_ocr(file_content)

                return full_text

            # Fallback if we get here
            return direct_text or "No text could be extracted from PDF"

    except Exception as e:
        print(f"Error during PDF text extraction: {e}")
        # Try OCR as last resort
        try:
            return extract_text_with_better_ocr(file_content)
        except Exception as ocr_error:
            print(f"OCR also failed: {ocr_error}")
            raise ValueError(f"Could not extract text from PDF: {str(e)}")


def extract_text_with_ocr(pdf_content: bytes, page_indices=None) -> str:
    """
    Standard OCR for PDF images - balanced speed and accuracy.

    Args:
        pdf_content: Binary PDF content
        page_indices: Optional list of page indices to process (0-based)

    Returns:
        Extracted text from images
    """
    try:
        # Convert PDF to images
        images = convert_from_bytes(
            pdf_content,
            dpi=300,  # Higher DPI for better quality on image PDFs
            fmt="jpeg",
            poppler_path=get_poppler_path(),
        )

        # Process only specific pages if requested
        if page_indices is not None and page_indices:
            if len(images) > 0 and max(page_indices) < len(images):
                images = [images[i] for i in page_indices]
            else:
                # Adjust for out of range indices
                valid_indices = [i for i in page_indices if i < len(images)]
                images = [images[i] for i in valid_indices]

        # Standard OCR config - tesseract 4+ preferred
        custom_config = r"--oem 3 --psm 6"  # More accurate for text in images

        # Process images
        text = ""
        for i, img in enumerate(images):
            try:
                # Convert to grayscale
                gray_img = img.convert("L")

                # Enhance contrast for better OCR
                enhancer = ImageEnhance.Contrast(gray_img)
                enhanced_img = enhancer.enhance(2.0)  # Stronger contrast for image PDFs

                # Extract text
                page_text = pytesseract.image_to_string(
                    enhanced_img, config=custom_config
                )

                if page_text.strip():
                    # Only add page number for multiple pages
                    if len(images) > 1:
                        page_num = page_indices[i] + 1 if page_indices else i + 1
                        text += f"[Page {page_num}]\n{page_text}\n\n"
                    else:
                        text += f"{page_text}\n\n"

            except Exception as page_error:
                print(f"Error on page {i}: {page_error}")
                continue

        return text
    except Exception as e:
        print(f"OCR processing error: {e}")
        raise


def extract_text_with_better_ocr(pdf_content: bytes) -> str:
    """
    Higher quality OCR for difficult images.

    Args:
        pdf_content: Binary PDF content

    Returns:
        Extracted text from images
    """
    try:
        # Use higher DPI
        images = convert_from_bytes(pdf_content, dpi=400, fmt="jpeg")  # Higher quality

        if not images:
            return ""

        # Try multiple OCR approaches (LSTM and Legacy)
        text = ""
        for i, img in enumerate(images):
            try:
                # Convert to RGB and then to grayscale
                if img.mode != "RGB":
                    img = img.convert("RGB")
                gray = img.convert("L")

                # Enhance contrast more aggressively
                enhancer = ImageEnhance.Contrast(gray)
                enhanced = enhancer.enhance(2.5)

                # Try both OCR engines and take the better result
                lstm_text = pytesseract.image_to_string(
                    enhanced, config="--oem 1 --psm 6"
                )
                neural_text = pytesseract.image_to_string(
                    enhanced, config="--oem 3 --psm 6"
                )

                # Use the longer result
                page_text = (
                    lstm_text if len(lstm_text) > len(neural_text) else neural_text
                )

                if len(images) > 1:
                    text += f"[Page {i+1}]\n{page_text}\n\n"
                else:
                    text += f"{page_text}\n\n"

            except Exception as page_error:
                print(f"Error processing page {i} with better OCR: {page_error}")

        return text
    except Exception as e:
        print(f"Enhanced OCR processing error: {e}")
        return ""


def get_poppler_path():
    """
    Get poppler path based on environment.
    Override this with your specific path if needed.
    """
    # Common locations - adjust as needed for your environment
    if os.name == "nt":  # Windows
        return None  # Set your Windows path if needed
    elif os.name == "posix":  # Linux/Mac
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

            # Convert to grayscale
            img = img.convert("L")

            # Enhance contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)

            # OCR config
            custom_config = r"--oem 3 --psm 6"  # More accurate for images
            text = pytesseract.image_to_string(img, config=custom_config)
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
    try:
        print(f"Processing file with content type: {content_type}")

        if "pdf" in content_type:
            text = extract_text_from_pdf(file_content)

            # PDF is special case - even minimal text is useful
            if not text or len(text.strip()) < 5:
                print(f"Warning: Insufficient text extracted from PDF")

                # One more attempt with higher quality OCR
                try:
                    text = extract_text_with_better_ocr(file_content)
                    if text.strip():
                        return text
                except:
                    pass

                raise ValueError("Could not extract meaningful text from file")

            return text

        elif (
            "docx" in content_type
            or "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            in content_type
        ):
            text = extract_text_from_docx(file_content)
            if not text or len(text.strip()) < 5:
                raise ValueError("Insufficient text extracted from DOCX file")
            return text

        elif "text/" in content_type:
            return file_content.decode("utf-8", errors="replace")

        elif "image/" in content_type:
            text = extract_text_from_image(file_content)
            if not text or len(text.strip()) < 5:
                raise ValueError("No text detected in image")
            return text

        else:
            raise ValueError(f"Unsupported content type: {content_type}")

    except Exception as e:
        print(f"Error in extract_text_from_file: {str(e)}")
        raise
