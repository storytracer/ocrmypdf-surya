"""
OCRmyPDF plugin for Surya OCR engine integration.

This plugin replaces Tesseract with Surya OCR in the OCRmyPDF pipeline,
providing superior recognition capabilities for a wide range of languages.
"""

from ocrmypdf import hookimpl

from .plugin import (
    get_ocr_engine,
    add_options,
    check_options,
    initialize,
    optimize_pdf,
    SuryaOcrEngine,
)

__version__ = '0.1.0'