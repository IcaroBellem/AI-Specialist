import os
import joblib
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import re

def split_text(text, max_tokens=512):
    sentences = re.split(r'(?<=[.!?]) +', text)
    segments = []
    current_segment = ""
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_tokens:
            segments.append(current_segment.strip())
            current_segment = sentence
            current_length = sentence_length
        else:
            current_segment += " " + sentence
            current_length += sentence_length
    
    if current_segment:
        segments.append(current_segment.strip())
    
    return segments

def preprocess_pdfs(pdf_paths, processed_dir):
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    
    embeddings_path = os.path.join(processed_dir, "embeddings.npy")
    documents_path = os.path.join(processed_dir, "documents.pkl")
    
    documents = []
    if os.path.exists(embeddings_path) and os.path.exists(documents_path):
        embeddings = np.load(embeddings_path)
        documents = joblib.load(documents_path)
    else:
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        for pdf_path in pdf_paths:
            reader = PdfReader(pdf_path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    print(f"Texto extraído da página {page_num+1} do PDF {pdf_path}")
                    segments = split_text(text, max_tokens=512)
                    documents.extend(segments)
        
        embeddings = model.encode(documents, convert_to_numpy=True)
        np.save(embeddings_path, embeddings)
        joblib.dump(documents, documents_path)

    return embeddings, documents

def is_valid_input(prompt):
    if re.search(r'[a-zA-Z0-9]', prompt) and len(prompt) > 2:
        return True
    return False
