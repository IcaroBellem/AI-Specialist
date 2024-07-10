import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
import re
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_google_genai import ChatGoogleGenerativeAI

# Carregar as variáveis de ambiente do arquivo .env
load_dotenv()

# Configurações do modelo
model = os.getenv("MODEL")

# Configuração do LLM
llm = ChatGoogleGenerativeAI(
    temperature=0.6,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    model=model,
    max_tokens=100,
)

# Função para dividir texto em segmentos menores
def split_text(text, max_tokens=500):
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

# Carregar e indexar os documentos PDF
pdf_paths = [
    "./data/CPU.pdf",
    "./data/Minicurso_Manutencao_ Computadores.pdf",
    "./data/Estrutra_do_computador.pdf",
    "./data/arquitetura_e_manutencao.pdf",
    "./data/usabilidade_basica.pdf",
    "./data/How2Build_PC.pdf",
    "./data/Icrim.pdf"
]

vectorizer = TfidfVectorizer()
documents = []

# Pré-processamento dos PDFs e criação dos embeddings
def preprocess_pdfs():
    for pdf_path in pdf_paths:
        print(f"Lendo PDF: {pdf_path}")
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                print(f"Texto extraído da página {page_num+1} do PDF {pdf_path}")
                segments = split_text(text)
                documents.extend(segments)
            else:
                print(f"Nenhum texto encontrado na página {page_num+1} do PDF {pdf_path}")
    if documents:
        embeddings = vectorizer.fit_transform(documents).toarray()
        # Salvar embeddings, documentos e vectorizer
        np.save("embeddings.npy", embeddings)
        joblib.dump(documents, "documents.pkl")
        joblib.dump(vectorizer, "vectorizer.pkl")
    else:
        print("Nenhum documento encontrado nos PDFs fornecidos.")

# Verificar se os arquivos pré-processados existem
if not os.path.exists("embeddings.npy") or not os.path.exists("documents.pkl") or not os.path.exists("vectorizer.pkl"):
    preprocess_pdfs()
else:
    # Carregar embeddings, documentos e vectorizer pré-processados
    embeddings = np.load("embeddings.npy")
    documents = joblib.load("documents.pkl")
    vectorizer = joblib.load("vectorizer.pkl")

# Função para buscar documentos relevantes
def similarity_search(query, k=5):
    # Verificar se o vetor TF-IDF está ajustado
    if not hasattr(vectorizer, 'vocabulary_'):
        preprocess_pdfs()
    query_embedding = vectorizer.transform([query]).toarray()
    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    indices = similarities.argsort()[-k:][::-1]
    return [documents[idx] for idx in indices]

# Função para interação com o chatbot
def chatbot_interaction(question):
    # Responder a perguntas básicas
    basic_responses = {
        "bom dia": "Bom dia! Como posso ajudar você hoje?",
        "boa tarde": "Boa tarde! Como posso ajudar você hoje?",
        "boa noite": "Boa noite! Como posso ajudar você hoje?",
        "como você está?": "Estou bem, obrigado! Como posso ajudar você hoje?",
        "qual o seu nome?": "Eu sou o TechzAI, seu assistente virtual especializado em hardware e informática.",
        "o que você faz?": "Eu sou um assistente virtual especializado em responder perguntas sobre hardware e informática. Como posso ajudar você hoje?"
    }
    
    if question in basic_responses:
        return basic_responses[question]

    # Buscar documentos relevantes usando embeddings
    docs = similarity_search(question, k=5)
    context = " ".join(docs)
    
    if context.strip():  # Se houver contexto relevante nos PDFs
        prompt = f"Você é um especialista em informática e hardware simpatico e está respondendo a pergunta de um usuário. O usuário pergunta: {question}. Responda a pergunta do usuário de forma amigável e informativa **somente e exclusivamente com os dados se caso for perguntas simples e basicas responda normalmente: {context}, jamais responda algo que não esteja nos dados fornecidos**. **Caso não saiba a resposta, diga que não sabe.**"
    else:
        prompt = f"Você é um assistente especializado em informática e hardware. Responda a pergunta do usuário da melhor forma possível. Pergunta: {question}"
    
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"Erro ao chamar API: {e}")
        return f"Desculpe, não consegui entender sua pergunta. Erro: {e}"

# Configuração da página do Streamlit
st.set_page_config(page_title="TechzAI", initial_sidebar_state="auto")
st.title("***TechzAI***")
st.markdown("*Soluções* **Inteligentes** ao seu ***Alcance!***.")

# Inicializar mensagens de sessão
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Olá! Como posso ajudar você hoje?"}]

# Exibir mensagens na interface do chat
for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])

# Função para validar a entrada do usuário
def is_valid_input(prompt):
    if re.search(r'[a-zA-Z0-9]', prompt) and len(prompt) > 2:
        return True
    return False

# Entrada do usuário
if question := st.chat_input("Qual é a sua dúvida hoje?"):
    question = question.strip().lower()
    if question and is_valid_input(question):
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        with st.spinner("Processando..."):
            try:
                response = chatbot_interaction(question)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.chat_message("assistant").write(response)
            except Exception as e:
                print(f"Erro ao chamar API: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Desculpe, não consegui entender sua pergunta. Erro: {e}"})
                st.chat_message("assistant").write("Desculpe, não consegui entender sua pergunta.")
    else:
        st.warning("Por favor, insira uma pergunta válida.")
