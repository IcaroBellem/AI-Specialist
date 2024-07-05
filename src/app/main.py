import os
import openai
import dotenv
import fitz
import streamlit as st
from io import BytesIO
import numpy as np
import re

dotenv.load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

openai.api_type = "azure"
openai.api_base = endpoint
openai.api_version = "2024-02-01"
openai.api_key = api_key

st.set_page_config(page_title="TechzAI", initial_sidebar_state="auto")

pdf_path = [
    "./data/CPU.pdf",
    "./data/Minicurso_Manutencao_ Computadores.pdf",	
    "./data/Estrutra_do_computador.pdf"
]
# Função para extrair texto do PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Função para dividir o texto em segmentos menores
def split_text_into_segments(text, max_tokens=300):
    words = text.split()
    segments = []
    current_segment = []

    for word in words:
        current_segment.append(word)
        if len(current_segment) >= max_tokens:
            segments.append(' '.join(current_segment))
            current_segment = []

    if current_segment:
        segments.append(' '.join(current_segment))
    
    return segments

# Função para criar embeddings
@st.cache_data
def create_embeddings(all_text_segments):
    embeddings = []
    for segment in all_text_segments:
        response = openai.Embedding.create(input=segment, engine=embedding_deployment)
        embeddings.append(response['data'][0]['embedding'])
    return embeddings

# Função para pesquisar com embeddings
def search_with_embeddings(query, embeddings, all_text_segments):
    response = openai.Embedding.create(input=query, engine=embedding_deployment)
    query_embedding = response['data'][0]['embedding']

    scores = [np.dot(query_embedding, emb) for emb in embeddings]
    
    # Ordenar segmentos por relevância
    sorted_indices = np.argsort(scores)[::-1]
    sorted_segments = [all_text_segments[i] for i in sorted_indices]
    
    # Selecionar os segmentos mais relevantes sem ultrapassar o limite de tokens
    selected_segments = []
    total_tokens = 0
    for segment in sorted_segments:
        segment_tokens = len(segment.split())
        if total_tokens + segment_tokens <= 6000:
            selected_segments.append(segment)
            total_tokens += segment_tokens
        else:
            break

    return ' '.join(selected_segments)

def load_and_process_pdfs(pdf_file_paths):
    all_text_segments = []
    for file_path in pdf_file_paths:
        text = extract_text_from_pdf(file_path)
        segments = split_text_into_segments(text, 30)
        all_text_segments.extend(segments)
    return all_text_segments
 
all_text_segments = load_and_process_pdfs(pdf_path)

# Respostas básicas
basic_responses = {
    "bom dia": "Bom dia! Como posso ajudar você hoje?",
    "boa tarde": "Boa tarde! Como posso ajudar você hoje?",
    "boa noite": "Boa noite! Como posso ajudar você hoje?",
    "como você está?": "Estou bem, obrigado por perguntar! E você?",
    "quem é você?": "Eu sou um assistente virtual criado para responder perguntas sobre hardware e informática em geral.",
}

st.title("***TechzAI***")
st.markdown("*Soluções* **Inteligentes** ao seu ***Alcançe!***.")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Olá! Como posso ajudar você hoje?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

def is_valid_input(prompt):
    if re.search(r'[a-zA-Z0-9]', prompt) and len(prompt) > 2:
        return True
    return False

embeddings = create_embeddings(all_text_segments)

if prompt := st.chat_input("Qual é a sua dúvida hoje?"):
    prompt = prompt.strip().lower()
    if prompt and is_valid_input(prompt):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        if prompt in basic_responses:
            response = basic_responses[prompt]
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
        else:
            with st.spinner("Processando..."):
                try:
                    relevant_text = search_with_embeddings(prompt, embeddings, all_text_segments)

                    response = openai.ChatCompletion.create(
                        engine=deployment,
                        messages=[
                            {"role": "system", "content": "Você é um assistente que responde perguntas com base na sua espcialidade (ser um assistente de hardware e informatica geral). Caso não seja com base nessa especialidade, ou caso voce nao entenda responda de varias formas(aleatoriamente) que nao consegue responder sobre o perguntado, e que o seu foco é responder perguntas relacionadas a hardware e informatica no geral. Use apenas o seguinte texto para elaborar boas respostas e responder à pergunta do usuário, caso seja do contexto e nao tenha no seguinte texto busque em outro lugar, voce também sempre continuara respondendo a conversa com base no contexto de sua especialidade: " + relevant_text},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=300,
                        n=1,
                        temperature=0.7,
                    )

                    msg = response['choices'][0]['message']['content']
                    st.session_state.messages.append({"role": "assistant", "content": msg})
                    st.chat_message("assistant").write(msg)
                except Exception as e:
                    print(f"Erro ao chamar API: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": "Desculpe, não consegui entender sua pergunta."})
                    st.chat_message("assistant").write("Desculpe, não consegui entender sua pergunta.")
    else:
        st.warning("Entrada inválida. Por favor, digite uma pergunta válida.")
