import os
import streamlit as st
import numpy as np
import joblib
from dotenv import load_dotenv
from utils import preprocess_pdfs, is_valid_input
from chatbot import chatbot_interaction
from config import load_config
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
from loading_animation import get_loading_animation

# Configuração da página
st.set_page_config(page_title="TechzAI", initial_sidebar_state="auto", page_icon="./assets/web-logo.ico")

load_dotenv()

# Diretórios e Inicializações
pdf_dir = "./data"
processed_dir = "./data_processed"
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

pdf_files = [
    "CPU.pdf",
    "Minicurso_Manutencao_ Computadores.pdf",
    "Estrutra_do_computador.pdf",
    "arquitetura_e_manutencao.pdf",
    "usabilidade_basica.pdf",
    "How2Build_PC.pdf",
    "Icrim.pdf",
    "defeitos e resoluções.pdf",
    "C#.pdf",
    "EXCEL TOTAL.pdf",
    "HTML e CSS.pdf",
    "História dos Teclados Mecânicos.pdf",
    "PC_atualized.pdf",
    "windows_avançado.pdf", 
    "GPU.pdf",  
    "guia_500_comandos_Linux.pdf",
    "formatação.pdf",
    "AngularBasico.pdf",
    "basic-i.pdf",
    "Aulas_SQL.pdf",
    "gasket mount.pdf"
]

pdf_paths = [os.path.join(pdf_dir, file) for file in pdf_files]

# Função para carregar embeddings e documentos
@st.cache_data
def load_embeddings_documents():
    embeddings, documents = preprocess_pdfs(pdf_paths, processed_dir)
    return embeddings, documents

embeddings, documents = load_embeddings_documents()

model, llm = load_config()

# UI da Página
st.title("***TechzAI***")
st.markdown("*Soluções* **Inteligentes** ao seu ***Alcance!***.")
st.sidebar.image("./assets/fundo-sidebar.png")

# Estado de Sessão para Conversas
if "conversations" not in st.session_state:
    st.session_state["conversations"] = [{"title": "Novo Chat", "messages": [{"role": "assistant", "content": "Olá! Como posso ajudar você hoje?"}]}]
if "delete_confirm" not in st.session_state:
    st.session_state["delete_confirm"] = None
if "current_conversation_index" not in st.session_state:
    st.session_state["current_conversation_index"] = 0

# Funções para Gerenciar Conversas
def add_new_conversation():
    new_title = f"Novo Chat {len(st.session_state['conversations']) + 1}"
    st.session_state["conversations"].append({"title": new_title, "messages": [{"role": "assistant", "content": "Olá! Como posso ajudar você hoje?"}]})
    st.session_state["current_conversation_index"] = len(st.session_state['conversations']) - 1

st.sidebar.title("Conversas")
if st.sidebar.button("Nova Conversa"):
    add_new_conversation()

conversation_titles = [conv["title"] for conv in st.session_state["conversations"]]
if conversation_titles:
    selected_conversation = st.sidebar.selectbox("Selecionar Conversa", conversation_titles, index=st.session_state["current_conversation_index"])
    st.session_state["current_conversation_index"] = conversation_titles.index(selected_conversation)
else:
    st.sidebar.write("Nenhuma conversa disponível.")

def edit_conversation_title(index, new_title):
    st.session_state["conversations"][index]["title"] = new_title

def delete_conversation(index):
    if st.session_state["delete_confirm"] == index:
        st.session_state["conversations"].pop(index)
        st.session_state["delete_confirm"] = None
        st.session_state["current_conversation_index"] = max(0, len(st.session_state["conversations"]) - 1)
    else:
        st.session_state["delete_confirm"] = index

for i, conv in enumerate(st.session_state["conversations"]):
    with st.sidebar.expander(conv["title"], expanded=False):
        new_title = st.text_input("Editar título", value=conv["title"], key=f"title_{i}")
        if st.button("Salvar título", key=f"save_title_{i}"):
            edit_conversation_title(i, new_title)
        if st.button("Apagar", key=f"delete_{i}"):
            delete_conversation(i)
        if st.session_state["delete_confirm"] == i:
            st.write("Tem certeza que deseja apagar esta conversa?")
            st.button("Sim, apagar", key=f"confirm_delete_{i}", on_click=delete_conversation, args=(i,))
            st.button("Não, cancelar", key=f"cancel_delete_{i}", on_click=lambda: st.session_state.update({"delete_confirm": None}))

# Exibir Conversas
if st.session_state["conversations"]:
    current_conv = st.session_state["conversations"][st.session_state["current_conversation_index"]]
else:
    current_conv = None

if current_conv:
    for message in current_conv["messages"]:
        st.chat_message(message["role"]).write(message["content"])
else:
    st.warning("Nenhuma conversa disponível. Por favor, inicie uma nova conversa na barra lateral.")

# Processamento em Background
def process_question(question, history):
    return chatbot_interaction(question, history, documents, embeddings, llm)

# Adicionar a animação de carregamento
loading_animation = get_loading_animation()
st.markdown(loading_animation, unsafe_allow_html=True)

# Entrada de Chat
if question := st.chat_input("Qual é a sua dúvida hoje?"):
    question = question.strip().lower()
    if question and is_valid_input(question):
        if current_conv is not None:
            history = current_conv["messages"]
            current_conv["messages"].append({"role": "user", "content": question})
            
            st.chat_message("user").write(question)
            
            # Mostrar animação de carregamento
            st.markdown('<style>#loading{display:block;}</style>', unsafe_allow_html=True)
            
            with ThreadPoolExecutor() as executor:
                future = executor.submit(process_question, question, history)
                response = future.result()
            
            # Esconder animação de carregamento
            st.markdown('<style>#loading{display:none;}</style>', unsafe_allow_html=True)
            
            current_conv["messages"].append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
        else:
            st.warning("Por favor, selecione uma conversa ou inicie uma nova.")
    else:
        st.warning("Por favor, insira uma pergunta válida.")
