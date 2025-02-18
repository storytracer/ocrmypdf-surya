"""
Surya OCR plugin for OCRmyPDF.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Sequence
from pkg_resources import get_distribution

import pluggy
from ocrmypdf import Executor, OcrEngine, PdfContext, hookimpl
from ocrmypdf._exec import tesseract
from ocrmypdf.builtin_plugins.optimize import optimize_pdf as default_optimize_pdf
from PIL import Image
from surya.recognition import RecognitionPredictor
from .converter import surya_to_hocr
from surya.detection import DetectionPredictor

log = logging.getLogger(__name__)

# Global predictors
_recognition_predictor = None
_detection_predictor = None

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
    global _recognition_predictor, _detection_predictor
    
    # Initialize predictors with suppressed output
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            _recognition_predictor = RecognitionPredictor()
            _detection_predictor = DetectionPredictor()


@hookimpl
def check_options(options):
    """Validate options and set up Surya environment variables."""
    options.jobs = 1
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
        # Use global predictors
        self.recognition_predictor = _recognition_predictor
        self.detection_predictor = _detection_predictor
        
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
        output_text: Path,
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
            
            # Suppress ALL output from surya
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    predictions = self.recognition_predictor(
                        [image], 
                        [surya_langs if surya_langs else None], 
                        self.detection_predictor
                    )
            
            if not predictions or not predictions[0]:
                raise RuntimeError("Surya OCR failed to produce results")
            
            # Get the first (and only) page from results
            page_data = predictions[0]
            
            # Extract plain text from hOCR for output_text
            if output_text:
                text = '\n'.join([line.text for line in page_data.text_lines])
                with open(output_text, 'w', encoding='utf-8') as f:
                    f.write(text)
            
            # Generate hOCR from Surya results
            hocr_content = surya_to_hocr(
                page_data,
                str(input_file),
                image_dpi,
                self.version()
            )
            
            # Write the hOCR content to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(hocr_content)
            
        except Exception as e:
            raise RuntimeError(f"Surya OCR failed: {str(e)}")

    def generate_hocr(self, input_file, output_hocr, output_text, options):
        """Generate hOCR output from an image file."""
        self.recognize(
            input_file=input_file,
            output_file=output_hocr,
            output_text=output_text,
            language=options.languages,
            image_dpi=options.image_dpi,
            options=options
        )

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
