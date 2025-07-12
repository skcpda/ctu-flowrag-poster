import os
import fitz  # PyMuPDF
import nltk
import numpy as np
from PIL import Image
import pytesseract
import io

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize, word_tokenize

# Settings
chunk_size = 512
pdf_folder = "."  # current directory

# Output vars
token_counts = []
sentence_counts = []
chunk_counts = []

def extract_text_with_ocr(page):
    """Extract text from a page using OCR if needed."""
    # First try regular text extraction
    text = page.get_text()
    
    # If no text found, try OCR
    if not text.strip():
        # Convert page to image
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 DPI
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Perform OCR
        text = pytesseract.image_to_string(img)
    
    return text

pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

for filename in pdf_files:
    print(f"Processing {filename}...")
    doc = fitz.open(os.path.join(pdf_folder, filename))
    full_text = ""
    
    for page_num, page in enumerate(doc):
       # print(f"  Processing page {page_num + 1}/{len(doc)}...")
        full_text += extract_text_with_ocr(page)

    sentences = sent_tokenize(full_text, language='english')
    tokens = word_tokenize(full_text)
    
    token_counts.append(len(tokens))
    sentence_counts.append(len(sentences))
    chunk_counts.append(len(tokens) // chunk_size)

# Aggregate stats
book_count = len(pdf_files)
total_tokens = sum(token_counts)
total_sentences = sum(sentence_counts)
mean_tokens = np.mean(token_counts)
std_tokens = np.std(token_counts)
total_chunks = sum(chunk_counts)

# Display results
print(f"\nResults:")
print(f"Book count: {book_count}")
print(f"Tokens: {total_tokens:,}")
print(f"Sentences: {total_sentences:,}")
print(f"Mean tokens/doc: {int(mean_tokens):,} ± {int(std_tokens):,}")
print(f"Chunks: {total_chunks:,}")
print("Licence: Unknown or not parsed from metadata")
