import os
import joblib
import numpy as np
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def split_text(text, vectorizer, max_tokens=500):
    sentences = text.split('. ')
    segments = []
    current_segment = ""
    current_length = 0
    for sentence in sentences:
        sentence_length = len(vectorizer.build_tokenizer()(sentence))
        if current_length + sentence_length > max_tokens:
            segments.append(current_segment)
            current_segment = sentence
            current_length = sentence_length
        else:
            current_segment += " " + sentence
            current_length += sentence_length
    if current_segment:
        segments.append(current_segment)
    return segments

def preprocess_pdfs(pdf_paths, documents, vectorizer, processed_dir):
    # Verificar se o diretório existe e criar se não existir
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    embeddings_path = os.path.join(processed_dir, "embeddings.npy")
    documents_path = os.path.join(processed_dir, "documents.pkl")
    if os.path.exists(embeddings_path) and os.path.exists(documents_path):
        # Se os arquivos já existirem, carregue-os
        embeddings = np.load(embeddings_path)
        documents.extend(joblib.load(documents_path))
    else:
        # Caso contrário, processe os PDFs
        for pdf_path in pdf_paths:
            reader = PdfReader(pdf_path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    segments = split_text(text, vectorizer)
                    documents.extend(segments)
        # Salvar embeddings, documentos e vectorizer
        embeddings = vectorizer.fit_transform(documents).toarray()
        np.save(embeddings_path, embeddings)
        joblib.dump(documents, documents_path)
        joblib.dump(vectorizer, os.path.join(processed_dir, "vectorizer.pkl"))
    return embeddings, documents

def is_valid_input(prompt):
    if re.search(r'[a-zA-Z0-9]', prompt) and len(prompt) > 2:
        return True
    return False 
