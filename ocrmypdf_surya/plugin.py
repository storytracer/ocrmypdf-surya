"""
Surya OCR plugin for OCRmyPDF.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Sequence
from pkg_resources import get_distribution

import pluggy
from ocrmypdf import Executor, OcrEngine, PdfContext, hookimpl
from ocrmypdf._exec import tesseract
from ocrmypdf.builtin_plugins.optimize import optimize_pdf as default_optimize_pdf
from lxml import etree
from PIL import Image
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

log = logging.getLogger(__name__)

# ISO_639_3 to Surya language code mapping
# Surya generally uses ISO 639-1 codes and some custom codes
ISO_639_3_TO_SURYA: dict[str, str] = {
    "eng": "en",
    "fra": "fr",
    "deu": "de"
}


@hookimpl
def initialize(plugin_manager: pluggy.PluginManager):
    """Initialize plugin and handle conflicts."""
    # Suppress other OCR engines or plugins if needed
    pass


@hookimpl
def check_options(options):
    """Validate options and set up Surya environment variables."""
    # Handle Surya environment variables
    if hasattr(options, 'surya_env') and options.surya_env:
        for env_var in options.surya_env.split(','):
            if '=' in env_var:
                key, value = env_var.strip().split('=', 1)
                os.environ[key] = value
    
    if hasattr(options, 'surya_batch_size') and options.surya_batch_size:
        os.environ['RECOGNITION_BATCH_SIZE'] = str(options.surya_batch_size)


@hookimpl
def optimize_pdf(
    input_pdf: Path,
    output_pdf: Path,
    context: PdfContext,
    executor: Executor,
    linearize: bool,
) -> tuple[Path, Sequence[str]]:
    """Clean up any Surya resources before optimizing PDF."""
    # Clean up any resources if needed
    
    # Use the default optimize function
    return default_optimize_pdf(
        input_pdf=input_pdf,
        output_pdf=output_pdf,
        context=context,
        executor=executor,
        linearize=linearize,
    )


@hookimpl
def add_options(parser):
    """Add Surya-specific command line arguments."""
    surya_group = parser.add_argument_group("Surya OCR", "Surya OCR options")
    surya_group.add_argument(
        '--surya-env',
        help="Environment variables for Surya OCR (comma separated KEY=value pairs)",
        type=str
    )
    surya_group.add_argument(
        '--surya-batch-size',
        help="Batch size for Surya recognition (default: auto)",
        type=int
    )
    surya_group.add_argument(
        '--surya-detection-batch-size',
        help="Batch size for Surya detection (default: auto)",
        type=int,
        dest="surya_detection_batch_size"
    )
    surya_group.add_argument(
        '--surya-torch-device',
        help="Torch device to use (cpu, cuda)",
        choices=['cpu', 'cuda'],
        default=None
    )


class SuryaOcrEngine(OcrEngine):
    """OCR engine that uses Surya OCR"""

    def __init__(self):
        """Initialize Surya predictors"""
        super().__init__()
        self.recognition_predictor = RecognitionPredictor()
        self.detection_predictor = DetectionPredictor()

    @staticmethod
    def version():
        """Return Surya version information as a string"""
        try:
            return get_distribution('surya-ocr').version
        except Exception:
            return "unknown"

    @staticmethod
    def creator_tag(options):
        """Return the creator tag for the PDF metadata."""
        tag = "-PDF" if options.pdf_renderer == "sandwich" else ""
        return f"Surya OCR{tag}"

    def __str__(self):
        return "Surya OCR"

    @staticmethod
    def languages(options):
        """Return list of languages supported by Surya"""
        # Return ISO 639-3 language codes that map to Surya languages
        return ISO_639_3_TO_SURYA.keys()

    @staticmethod
    def get_orientation(input_file: Path, options):
        """Get the orientation of the image.
        
        Surya doesn't provide orientation detection, so we fall back to Tesseract
        for this functionality only.
        """
        # Fall back to Tesseract for orientation detection
        return tesseract.get_orientation(
            input_file,
            engine_mode=options.tesseract_oem,
            timeout=options.tesseract_non_ocr_timeout,
        )

    def recognize(
        self,
        input_file: Path,
        output_file: Path,
        language: List[str],
        image_dpi: Optional[float],
        options: Dict[str, Any],
    ):
        """Run Surya OCR on an image and output hOCR"""
        # Set up torch device if specified
        if hasattr(options, 'surya_torch_device') and options.surya_torch_device:
            os.environ['TORCH_DEVICE'] = options.surya_torch_device
            
        # Set detection batch size if specified
        if hasattr(options, 'surya_detection_batch_size') and options.surya_detection_batch_size:
            os.environ['DETECTOR_BATCH_SIZE'] = str(options.surya_detection_batch_size)
        
        # Map ISO 639-3 language codes to Surya language codes
        surya_langs = []
        for lang in language:
            if lang in ISO_639_3_TO_SURYA:
                surya_langs.append(ISO_639_3_TO_SURYA[lang])
            else:
                log.warning(f"Unsupported language code: {lang}")
        
        try:
            # Load image
            image = Image.open(input_file)
            
            # Run Surya OCR directly
            predictions = self.recognition_predictor(
                [image], 
                [surya_langs if surya_langs else None], 
                self.detection_predictor
            )
            
            if not predictions or not predictions[0]:
                raise RuntimeError("Surya OCR failed to produce results")
            
            # Get the first (and only) page from results
            page_data = predictions[0]
            
            # Generate hOCR from Surya results
            self._surya_to_hocr(page_data, input_file, output_file, image_dpi)
            
        except Exception as e:
            raise RuntimeError(f"Surya OCR failed: {str(e)}")
    
    def _surya_to_hocr(self, page_data, input_file, output_file, image_dpi):
        """Convert Surya OCR results to hOCR format"""
        source_bbox = page_data.image_bbox
        target_bbox = source_bbox  # Use same dimensions
        
        # Create hOCR document
        xml_ns = "http://www.w3.org/XML/1998/namespace"
        
        # Create the root element
        root = etree.Element(
            "html",
            attrib={
                "xmlns": "http://www.w3.org/1999/xhtml",
                "{%s}lang" % xml_ns: "en",
                "lang": "en",
            },
        )
        
        # Create head section
        head = etree.SubElement(root, "head")
        title = etree.SubElement(head, "title")
        title.text = "OCR Results"
        
        etree.SubElement(
            head,
            "meta",
            attrib={"http-equiv": "Content-Type", "content": "text/html; charset=utf-8"},
        )
        etree.SubElement(
            head, 
            "meta", 
            attrib={"name": "ocr-system", "content": f"Surya OCR {SuryaOcrEngine.version()}"}
        )
        etree.SubElement(
            head,
            "meta",
            attrib={
                "name": "ocr-capabilities",
                "content": "ocr_page ocr_carea ocr_par ocr_line ocrx_word",
            },
        )
        etree.SubElement(
            head,
            "meta",
            attrib={
                "name": "ocr-number-of-pages",
                "content": "1",
            },
        )
        
        body = etree.SubElement(root, "body")
        
        # Create page element with REQUIRED dimensions
        # The page title must include bbox coordinates and page size in pixels
        dpi_str = f"{image_dpi}" if image_dpi else "300"
        image_dpi_float = float(dpi_str)
        
        # Calculate dimensions in pixels based on bbox
        width_px = int(target_bbox[2])
        height_px = int(target_bbox[3])
        
        ocr_page = etree.SubElement(
            body,
            "div",
            attrib={
                "class": "ocr_page",
                "id": "page_1",
                "title": (
                    f'image "{os.path.basename(input_file)}"; '
                    f'bbox 0 0 {width_px} {height_px}; '
                    f'ppageno 0; '
                    f'scan_res {dpi_str} {dpi_str}; '
                    f'size {width_px} {height_px}'
                )
            },
        )
        
        # Process text lines
        for text_line in page_data.text_lines:
            bbox = text_line.bbox
            bbox_str = f"{int(bbox[0])} {int(bbox[1])} {int(bbox[2])} {int(bbox[3])}"
            
            # Create the ocr_carea
            div_ocr_carea = etree.SubElement(
                ocr_page,
                "div",
                attrib={
                    "class": "ocr_carea",
                    "title": f"bbox {bbox_str}",
                },
            )
            
            # Create ocr_par
            p = etree.SubElement(
                div_ocr_carea,
                "p",
                attrib={
                    "class": "ocr_par",
                    "title": f"bbox {bbox_str}",
                },
            )
            
            # Create ocr_line with explicit baseline
            line = etree.SubElement(
                p,
                "span",
                attrib={
                    "class": "ocr_line",
                    "title": f"bbox {bbox_str}; baseline 0 0; x_size 30; x_descenders 7; x_ascenders 7",
                },
            )
            
            # Create word with text and confidence
            confidence = int((text_line.confidence or 0.0) * 100)
            word = etree.SubElement(
                line,
                "span",
                attrib={
                    "class": "ocrx_word",
                    "title": f"bbox {bbox_str}; x_wconf {confidence}",
                },
            )
            
            # Set text content
            word.text = text_line.text
        
        # Write the hOCR file
        doctype = '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">'
        
        with open(output_file, 'wb') as f:
            f.write(etree.tostring(
                root, 
                pretty_print=True, 
                encoding="utf-8", 
                doctype=doctype, 
                xml_declaration=True
            ))

    def generate_hocr(self, input_file, output_hocr, output_text, options):
        """Generate hOCR output from an image file."""
        self.recognize(
            input_file=input_file,
            output_file=output_hocr,
            language=options.languages,
            image_dpi=options.image_dpi,
            options=options
        )
        
        # Extract plain text from hOCR for output_text
        if output_text:
            tree = etree.parse(str(output_hocr))
            text_elements = tree.xpath('//span[@class="ocrx_word"]')
            text = ' '.join(elem.text for elem in text_elements if elem.text)
            output_text.write_text(text)

    def generate_pdf(self, input_file, output_pdf, output_text, options):
        """
        OCRmyPDF calls this method when pdf_renderer='sandwich'.
        For Surya, we don't implement this directly but rely on OCRmyPDF's
        sandwich renderer which uses our hOCR output.
        """
        # Generate hOCR which OCRmyPDF will then use for PDF creation
        hocr_temp = Path(tempfile.mktemp(suffix='.hocr'))
        try:
            self.generate_hocr(input_file, hocr_temp, output_text, options)
            # OCRmyPDF will handle converting hOCR to PDF
        finally:
            # Clean up temp file
            if hocr_temp.exists():
                hocr_temp.unlink()


@hookimpl
def get_ocr_engine():
    return SuryaOcrEngine()
