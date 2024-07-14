import os
import joblib
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import re
from concurrent.futures import ThreadPoolExecutor

def split_text(text, max_tokens=512):
    sentences = re.split(r'(?<=[.!?]) +', text)
    segments = []
    current_segment = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_tokens:
            segments.append(' '.join(current_segment).strip())
            current_segment = [sentence]
            current_length = sentence_length
        else:
            current_segment.append(sentence)
            current_length += sentence_length
    
    if current_segment:
        segments.append(' '.join(current_segment).strip())
    
    return segments

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    all_text = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            print(f"Texto extraído da página {page_num + 1} do PDF {pdf_path}")
            segments = split_text(text, max_tokens=512)
            all_text.extend(segments)
    return all_text

def preprocess_pdfs(pdf_paths, processed_dir):
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    
    embeddings_path = os.path.join(processed_dir, "embeddings.npy")
    documents_path = os.path.join(processed_dir, "documents.pkl")
    
    if os.path.exists(embeddings_path) and os.path.exists(documents_path):
        embeddings = np.load(embeddings_path)
        documents = joblib.load(documents_path)
    else:
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(extract_text_from_pdf, pdf_paths))
        
        documents = [segment for doc in results for segment in doc]
        embeddings = model.encode(documents, convert_to_numpy=True)
        
        np.save(embeddings_path, embeddings)
        joblib.dump(documents, documents_path)

    return embeddings, documents

def is_valid_input(prompt):
    return bool(re.search(r'[a-zA-Z0-9]', prompt)) and len(prompt) > 2
