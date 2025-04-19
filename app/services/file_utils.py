import io
import tempfile
from typing import Any, Dict, List, Optional

import fitz
import pdfplumber
from PIL import Image, ImageEnhance

# For OCR functionality
try:
    import pytesseract

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import docx

    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


class PDFExtractor:
    """A class to extract text, images, tables and links from PDF files"""

    def __init__(self, pdf_path):
        """Initialize with the path to the PDF file"""
        self.pdf_path = pdf_path
        self.extracted_text = None
        self.extracted_images = []
        self.extracted_tables = []
        self.extracted_links = []
        self.ocr_text = []  # Store OCR results from images
        self.page_count = 0
        self.metadata = {}

    def extract_all(self):
        """Extract all content types from the PDF"""
        print("Extracting all content from PDF...")
        self.extract_text()
        self.extract_images()
        self.extract_tables()
        self.extract_links()

        return {
            "text": self.extracted_text,
            "images": self.extracted_images,
            "tables": self.extracted_tables,
            "links": self.extracted_links,
            "ocr_text": self.ocr_text,
            "page_count": self.page_count,
            "metadata": self.metadata,
        }

    def extract_text(self):
        """Extract text from the PDF using PyMuPDF and fall back to OCR if needed"""
        print("Extracting text...")

        # Open the PDF
        doc = fitz.open(self.pdf_path)
        self.page_count = doc.page_count
        self.metadata = doc.metadata

        # Extract text from all pages
        full_text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            full_text += page_text

        # Check if we got meaningful text content
        if len(full_text.strip()) < 100 and OCR_AVAILABLE:
            print("Limited text found. Trying OCR on pages...")
            # Use OCR as fallback for pages with little text
            ocr_text = ""
            for page_num in range(
                min(doc.page_count, 10)
            ):  # Limit to first 10 pages for speed
                print(f"OCR processing page {page_num+1}...")
                page_text = self.extract_text_from_page_image(page_num)
                ocr_text += f"\n--- PAGE {page_num+1} ---\n"
                ocr_text += page_text

            # If OCR found text, use it
            if len(ocr_text.strip()) > len(full_text.strip()):
                full_text = ocr_text
                print(f"OCR extracted {len(full_text)} characters of text")
            else:
                print(
                    f"Extracted {len(full_text)} characters of text from {doc.page_count} pages"
                )
        else:
            print(
                f"Extracted {len(full_text)} characters of text from {doc.page_count} pages"
            )

        self.extracted_text = full_text
        doc.close()
        return full_text

    def extract_images(self):
        """Extract images from the PDF using PyMuPDF"""
        print("Extracting images...")

        doc = fitz.open(self.pdf_path)
        images = []

        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)

            if image_list:
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)

                        if base_image:
                            image_info = {
                                "page": page_num + 1,
                                "index": img_index,
                                "format": base_image.get("ext", "unknown"),
                                "width": base_image.get("width", 0),
                                "height": base_image.get("height", 0),
                                "data": base_image.get("image", None),
                            }

                            # Perform OCR if available and image is suitable
                            if OCR_AVAILABLE and base_image.get("image"):
                                try:
                                    # Extract text from image using OCR
                                    ocr_result = self.perform_ocr_on_image(
                                        base_image["image"]
                                    )
                                    if ocr_result:
                                        image_info["ocr_text"] = ocr_result
                                        # Add to the OCR results list
                                        self.ocr_text.append(
                                            {
                                                "page": page_num + 1,
                                                "image_index": img_index,
                                                "text": ocr_result,
                                            }
                                        )
                                except Exception as e:
                                    print(f"OCR error: {str(e)}")

                            images.append(image_info)
                    except Exception as e:
                        print(
                            f"Error processing image {img_index} on page {page_num+1}: {e}"
                        )

        self.extracted_images = images
        print(f"Extracted {len(images)} images")
        doc.close()
        return images

    def perform_ocr_on_image(self, image_data):
        """Perform OCR on an image to extract text"""
        if not OCR_AVAILABLE:
            return None

        try:
            # Convert image data to PIL Image
            pil_image = Image.open(io.BytesIO(image_data))

            # Convert to RGB if needed (some formats may cause issues)
            if pil_image.mode not in ["RGB", "L"]:
                pil_image = pil_image.convert("RGB")

            # Perform OCR
            text = pytesseract.image_to_string(pil_image)

            # Return extracted text if not empty
            if text and text.strip():
                return text.strip()

        except Exception as e:
            print(f"OCR error: {str(e)}")

        return None

    def extract_tables(self):
        """Extract tables from the PDF using pdfplumber"""
        print("Extracting tables...")

        with pdfplumber.open(self.pdf_path) as pdf:
            tables = []

            for page_num, page in enumerate(pdf.pages):
                try:
                    page_tables = page.extract_tables()

                    if page_tables:
                        for table_index, table in enumerate(page_tables):
                            # Process and clean the table
                            if table and len(table) > 0:
                                # Create proper column headers if they exist
                                headers = table[0] if table else None
                                data = table[1:] if table and len(table) > 1 else []

                                # Fix table data with robust error handling
                                fixed_table = self._fix_table_data(headers, data)

                                # Create a table info dictionary
                                table_info = {
                                    "page": page_num + 1,
                                    "index": table_index,
                                    "headers": fixed_table["headers"],
                                    "data": fixed_table["data"],
                                    "rows": len(fixed_table["data"]),
                                    "columns": (
                                        len(fixed_table["headers"])
                                        if fixed_table["headers"]
                                        else 0
                                    ),
                                }

                                tables.append(table_info)
                except Exception as e:
                    print(f"Error extracting tables from page {page_num+1}: {e}")

            self.extracted_tables = tables
            print(f"Extracted {len(tables)} tables")

        return tables

    def _fix_table_data(self, headers, data):
        """Fix common table issues with headers and data"""
        # Fix headers: replace empty with generic names and ensure uniqueness
        fixed_headers = []
        for i, h in enumerate(headers):
            # Clean the header
            header = str(h).strip() if h is not None else ""

            # Replace empty headers with generic names
            if header == "":
                header = f"Column_{i+1}"

            # Replace problematic characters for pandas
            header = header.replace("\n", " ").replace("\r", "")

            # Ensure uniqueness
            base_header = header
            counter = 1
            while header in fixed_headers:
                header = f"{base_header}_{counter}"
                counter += 1

            fixed_headers.append(header)

        # Fix data: ensure consistent row lengths
        fixed_data = []
        for row in data:
            # Clean row data
            cleaned_row = [
                str(cell).strip() if cell is not None else "" for cell in row
            ]

            # Ensure row has the same length as headers
            if len(cleaned_row) < len(fixed_headers):
                cleaned_row.extend([""] * (len(fixed_headers) - len(cleaned_row)))
            elif len(cleaned_row) > len(fixed_headers):
                cleaned_row = cleaned_row[: len(fixed_headers)]

            fixed_data.append(cleaned_row)

        return {"headers": fixed_headers, "data": fixed_data}

    def extract_links(self):
        """Extract links from the PDF using PyMuPDF"""
        print("Extracting links...")

        doc = fitz.open(self.pdf_path)
        links = []

        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            page_links = page.get_links()

            if page_links:
                for link_index, link in enumerate(page_links):
                    link_info = {
                        "page": page_num + 1,
                        "index": link_index,
                        "type": None,
                        "target": None,
                    }

                    if "uri" in link:
                        link_info["type"] = "external"
                        link_info["target"] = link["uri"]
                    elif "page" in link:
                        link_info["type"] = "internal"
                        link_info["target"] = f"Page {link['page']+1}"

                    links.append(link_info)

        self.extracted_links = links
        print(f"Extracted {len(links)} links")
        doc.close()
        return links

    def render_page_as_image(self, page_num, zoom=2):
        """Render a specific page as an image with a white background"""
        doc = fitz.open(self.pdf_path)
        page = doc.load_page(page_num)

        # Set a white background to avoid black/transparent background
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img_data = Image.open(io.BytesIO(pix.tobytes("png")))

        # Create a white background image
        white_bg = Image.new("RGB", img_data.size, (255, 255, 255))
        if img_data.mode == "RGBA":
            # Paste the image onto the white background, respecting alpha channel
            white_bg.paste(img_data, (0, 0), img_data)
            img_data = white_bg
        else:
            white_bg.paste(img_data, (0, 0))
            img_data = white_bg

        doc.close()
        return img_data

    def extract_text_from_page_image(self, page_num, zoom=2):
        """Extract text from a page image using OCR"""
        if not OCR_AVAILABLE:
            return "OCR not available. Please install pytesseract."

        try:
            # Get page as image
            img = self.render_page_as_image(page_num, zoom)

            # Perform OCR
            text = pytesseract.image_to_string(img)
            return text
        except Exception as e:
            print(f"OCR error on page {page_num}: {e}")
            return f"OCR error: {str(e)}"


class PDFTextExtractor:
    """A class to extract text, images, tables and links from PDF content bytes"""

    def __init__(self, file_content: bytes):
        """Initialize with the PDF file content as bytes"""
        self.file_content = file_content
        self.extracted_text = None
        self.extracted_images = []
        self.extracted_tables = []
        self.extracted_links = []
        self.ocr_text = []  # Store OCR results from images
        self.page_count = 0
        self.metadata = {}

    def extract_text(self) -> Dict[str, Any]:
        """
        Extract text, images, tables and links from PDF

        Returns:
            Dict containing extracted text, images, tables, links and metadata
        """
        extracted_result = {
            "text": "",
            "images": [],
            "tables": [],
            "links": [],
            "ocr_text": [],
            "metadata": {},
            "page_count": 0,
        }

        try:
            # Open the PDF directly from memory buffer
            with io.BytesIO(self.file_content) as memory_buffer:
                # Open the PDF with PyMuPDF from the memory buffer
                doc = fitz.open(stream=memory_buffer, filetype="pdf")

                self.page_count = doc.page_count
                self.metadata = doc.metadata

                extracted_result["page_count"] = self.page_count
                extracted_result["metadata"] = self.metadata

                # Extract text from all pages with page number tracking
                text_by_page = {}
                full_text = ""
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    page_text = page.get_text()
                    text_by_page[page_num] = page_text
                    full_text += page_text

                # If text content is limited, try OCR
                if len(full_text.strip()) < 100 and OCR_AVAILABLE:
                    print("Limited text found. Trying OCR on pages...")
                    # Use OCR as fallback for pages with little text
                    ocr_text = ""
                    for page_num in range(
                        min(doc.page_count, 10)
                    ):  # Limit to first 10 pages for speed
                        print(f"OCR processing page {page_num+1}...")
                        page_text = self.extract_text_from_page_image(doc, page_num)
                        ocr_text += f"\n--- PAGE {page_num+1} ---\n"
                        ocr_text += page_text

                    # If OCR found text, use it
                    if len(ocr_text.strip()) > len(full_text.strip()):
                        full_text = ocr_text
                        print(f"OCR extracted {len(full_text)} characters of text")
                    else:
                        print(
                            f"Extracted {len(full_text)} characters of text from {doc.page_count} pages"
                        )
                else:
                    print(
                        f"Extracted {len(full_text)} characters of text from {doc.page_count} pages"
                    )

                # Extract images with OCR text
                self.extracted_images = self.extract_images(doc)
                extracted_result["images"] = self.extracted_images

                # Extract links
                self.extracted_links = self.extract_links(doc)
                extracted_result["links"] = self.extracted_links

                # Close the document
                doc.close()

                # Extract tables with pdfplumber
                self.extracted_tables = self.extract_tables()
                extracted_result["tables"] = self.extracted_tables

                # Store text by page in the result
                extracted_result["text_by_page"] = text_by_page
                extracted_result["text"] = full_text.strip()
                extracted_result["ocr_text"] = self.ocr_text

                return extracted_result

        except Exception as e:
            print(f"Error extracting content from PDF: {e}")
            import traceback

            traceback.print_exc()

            # Return empty result with error info
            extracted_result["error"] = str(e)
            return extracted_result

    def extract_images(self, doc=None) -> List[Dict[str, Any]]:
        """Extract images from the PDF using PyMuPDF"""
        print("Extracting images...")

        images = []
        close_doc = False

        try:
            # If no document is provided, open it
            if doc is None:
                with io.BytesIO(self.file_content) as memory_buffer:
                    doc = fitz.open(stream=memory_buffer, filetype="pdf")
                    close_doc = True

            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                image_list = page.get_images(full=True)

                if image_list:
                    for img_index, img in enumerate(image_list):
                        try:
                            xref = img[0]
                            base_image = doc.extract_image(xref)

                            if base_image:
                                image_info = {
                                    "page": page_num + 1,
                                    "index": img_index,
                                    "format": base_image.get("ext", "unknown"),
                                    "width": base_image.get("width", 0),
                                    "height": base_image.get("height", 0),
                                    "data": base_image.get("image", None),
                                }

                                # Perform OCR if available and image is suitable
                                if OCR_AVAILABLE and base_image.get("image"):
                                    try:
                                        # Extract text from image using OCR
                                        ocr_result = self.perform_ocr_on_image(
                                            base_image["image"]
                                        )
                                        if ocr_result:
                                            image_info["ocr_text"] = ocr_result
                                            # Add to the OCR results list
                                            self.ocr_text.append(
                                                {
                                                    "page": page_num + 1,
                                                    "image_index": img_index,
                                                    "text": ocr_result,
                                                }
                                            )
                                    except Exception as e:
                                        print(f"OCR error: {str(e)}")

                                images.append(image_info)
                        except Exception as e:
                            print(
                                f"Error processing image {img_index} on page {page_num+1}: {e}"
                            )

            if close_doc:
                doc.close()

            print(f"Extracted {len(images)} images")
            return images

        except Exception as e:
            print(f"Error extracting images: {e}")
            if close_doc and doc:
                doc.close()
            return []

    def extract_tables(self) -> List[Dict[str, Any]]:
        """Extract tables from the PDF using pdfplumber"""
        print("Extracting tables...")
        tables = []

        try:
            # Create a temporary file for pdfplumber (it works better with files)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(self.file_content)
                temp_path = temp_file.name

            with pdfplumber.open(temp_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_tables = page.extract_tables()

                        if page_tables:
                            for table_index, table in enumerate(page_tables):
                                # Process and clean the table
                                if table and len(table) > 0:
                                    # Create proper column headers if they exist
                                    headers = table[0] if table else None
                                    data = table[1:] if table and len(table) > 1 else []

                                    # Fix table data with robust error handling
                                    fixed_table = self._fix_table_data(headers, data)

                                    # Create a table info dictionary
                                    table_info = {
                                        "page": page_num + 1,
                                        "index": table_index,
                                        "headers": fixed_table["headers"],
                                        "data": fixed_table["data"],
                                        "rows": len(fixed_table["data"]),
                                        "columns": (
                                            len(fixed_table["headers"])
                                            if fixed_table["headers"]
                                            else 0
                                        ),
                                    }

                                    tables.append(table_info)
                    except Exception as e:
                        print(f"Error extracting tables from page {page_num+1}: {e}")

            # Clean up the temporary file
            import os

            if os.path.exists(temp_path):
                os.unlink(temp_path)

            print(f"Extracted {len(tables)} tables")
            return tables

        except Exception as e:
            print(f"Error in table extraction: {e}")
            return []

    def _fix_table_data(self, headers, data):
        """Fix common table issues with headers and data"""
        # Fix headers: replace empty with generic names and ensure uniqueness
        fixed_headers = []
        for i, h in enumerate(headers):
            # Clean the header
            header = str(h).strip() if h is not None else ""

            # Replace empty headers with generic names
            if header == "":
                header = f"Column_{i+1}"

            # Replace problematic characters
            header = header.replace("\n", " ").replace("\r", "")

            # Ensure uniqueness
            base_header = header
            counter = 1
            while header in fixed_headers:
                header = f"{base_header}_{counter}"
                counter += 1

            fixed_headers.append(header)

        # Fix data: ensure consistent row lengths
        fixed_data = []
        for row in data:
            # Clean row data
            cleaned_row = [
                str(cell).strip() if cell is not None else "" for cell in row
            ]

            # Ensure row has the same length as headers
            if len(cleaned_row) < len(fixed_headers):
                cleaned_row.extend([""] * (len(fixed_headers) - len(cleaned_row)))
            elif len(cleaned_row) > len(fixed_headers):
                cleaned_row = cleaned_row[: len(fixed_headers)]

            fixed_data.append(cleaned_row)

        return {"headers": fixed_headers, "data": fixed_data}

    def extract_links(self, doc=None) -> List[Dict[str, Any]]:
        """Extract links from the PDF using PyMuPDF"""
        print("Extracting links...")

        links = []
        close_doc = False

        try:
            # If no document is provided, open it
            if doc is None:
                with io.BytesIO(self.file_content) as memory_buffer:
                    doc = fitz.open(stream=memory_buffer, filetype="pdf")
                    close_doc = True

            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                page_links = page.get_links()

                if page_links:
                    for link_index, link in enumerate(page_links):
                        link_info = {
                            "page": page_num + 1,
                            "index": link_index,
                            "type": None,
                            "target": None,
                        }

                        if "uri" in link:
                            link_info["type"] = "external"
                            link_info["target"] = link["uri"]
                        elif "page" in link:
                            link_info["type"] = "internal"
                            link_info["target"] = f"Page {link['page']+1}"

                        links.append(link_info)

            if close_doc:
                doc.close()

            print(f"Extracted {len(links)} links")
            return links

        except Exception as e:
            print(f"Error extracting links: {e}")
            if close_doc and doc:
                doc.close()
            return []

    def render_page_as_image(self, doc, page_num: int, zoom: int = 2) -> Image.Image:
        """Render a specific page as an image with a white background"""
        try:
            page = doc.load_page(page_num)

            # Set a white background to avoid black/transparent background
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            img_data = Image.open(io.BytesIO(pix.tobytes("png")))

            # Create a white background image
            white_bg = Image.new("RGB", img_data.size, (255, 255, 255))
            if img_data.mode == "RGBA":
                # Paste the image onto the white background, respecting alpha channel
                white_bg.paste(img_data, (0, 0), img_data)
                img_data = white_bg
            else:
                white_bg.paste(img_data, (0, 0))
                img_data = white_bg

            return img_data
        except Exception as e:
            print(f"Error rendering page {page_num} as image: {e}")
            # Return an empty white image
            return Image.new("RGB", (100, 100), (255, 255, 255))

    def extract_text_from_page_image(self, doc, page_num: int, zoom: int = 2) -> str:
        """Extract text from a page image using OCR"""
        if not OCR_AVAILABLE:
            return "OCR not available. Please install pytesseract."

        try:
            # Get page as image
            img = self.render_page_as_image(doc, page_num, zoom)

            # Perform OCR
            text = pytesseract.image_to_string(img)
            return text
        except Exception as e:
            print(f"OCR error on page {page_num}: {e}")
            return f"OCR error: {str(e)}"

    def perform_ocr_on_image(self, image_data: bytes) -> Optional[str]:
        """Perform OCR on an image to extract text"""
        if not OCR_AVAILABLE:
            return None

        try:
            # Convert image data to PIL Image
            pil_image = Image.open(io.BytesIO(image_data))

            # Convert to RGB if needed (some formats may cause issues)
            if pil_image.mode not in ["RGB", "L"]:
                pil_image = pil_image.convert("RGB")

            # Perform OCR
            text = pytesseract.image_to_string(pil_image)

            # Return extracted text if not empty
            if text and text.strip():
                return text.strip()

        except Exception as e:
            print(f"OCR error: {str(e)}")

        return None

    def extract_text_with_ocr(self) -> str:
        """
        Full OCR fallback for entire PDF using PyMuPDF to render pages
        """
        if not OCR_AVAILABLE:
            return "OCR not available. Please install pytesseract."

        try:
            # Open PDF directly from memory buffer
            with io.BytesIO(self.file_content) as memory_buffer:
                doc = fitz.open(stream=memory_buffer, filetype="pdf")

                text = ""
                # Process only the first 10 pages for speed
                pages_to_process = min(doc.page_count, 10)
                for page_num in range(pages_to_process):
                    page_text = self.extract_text_from_page_image(doc, page_num)
                    text += f"--- PAGE {page_num+1} ---\n"
                    text += page_text + "\n\n"

                doc.close()
                return text.strip()

        except Exception as e:
            print(f"Error during OCR processing: {e}")
            return ""

    def get_extraction_results(self) -> Dict[str, Any]:
        """Get all extraction results as a dictionary"""
        return {
            "metadata": self.metadata,
            "page_count": self.page_count,
            "images": self.extracted_images,
            "tables": self.extracted_tables,
            "links": self.extracted_links,
        }


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

    For PDF files, this will only extract the text, not images or tables.
    For full PDF processing, use PDFTextExtractor directly.
    """
    extractor = None

    if "pdf" in content_type:
        extractor = PDFTextExtractor(file_content)
        # For direct text extraction, just get the text from the full extract
        extraction_result = extractor.extract_text()
        return extraction_result.get("text", "")
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

    if extractor and not isinstance(extractor, PDFTextExtractor):
        return extractor.extract_text()
    return ""
    return ""
