import uuid
from typing import Tuple
import requests
from docling.document_converter import DocumentConverter
from fastapi import HTTPException
import os

def download_pdf(url: str) -> Tuple[str, str]:
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException:
        raise HTTPException(status_code=400, detail="Failed to download PDF")

    document_id = str(uuid.uuid4())
    pdf_path = f"temp/{document_id}.pdf"
    os.makedirs('temp', exist_ok=True)
    with open(pdf_path, "wb") as f:
        f.write(response.content)
    return pdf_path, document_id

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        converter = DocumentConverter()
        converter_output = converter.convert(pdf_path)
        text_pieces = []
        assembled_data = converter_output.assembled
        body_elements = assembled_data.body  # List of TextElement objects
        for element in body_elements:
            text = element.text  # Access the 'text' attribute
            if text:
                text_pieces.append(text)
        doc_text = '\n'.join(text_pieces)

        if not doc_text.strip():
            raise HTTPException(status_code=400, detail="The document contains no text")

        return doc_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
