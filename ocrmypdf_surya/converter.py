"""
Convert Surya OCR results to hOCR format.
"""

from lxml import etree
import os
import math
from typing import Optional


def surya_to_hocr(
    page_data,
    input_file: str,
    image_dpi: Optional[float],
    ocr_system_version: str
) -> str:
    """Convert Surya OCR results to hOCR format string.
    
    Args:
        page_data: Surya OCR page results containing text lines and bounding boxes
        input_file: Name of the input image file
        image_dpi: DPI of the input image
        ocr_system_version: Version string of the OCR system
        
    Returns:
        str: Complete hOCR document as a string
    """
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
        attrib={"name": "ocr-system", "content": f"Surya OCR {ocr_system_version}"}
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
        confidence_val = text_line.confidence
        if confidence_val is None or (isinstance(confidence_val, float) and math.isnan(confidence_val)):
            confidence = 0
        else:
            confidence = int(float(confidence_val) * 100)
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
    
    # Generate the complete hOCR document as a string
    doctype = '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">'
    
    return etree.tostring(
        root,
        pretty_print=True,
        encoding="utf-8",
        doctype=doctype,
        xml_declaration=True
    ).decode('utf-8')
