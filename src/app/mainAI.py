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
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    model=model,
    max_tokens=300,
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
    "./data/Icrim.pdf",
    "./data/defeitos e resoluções.pdf"
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
def chatbot_interaction(question, history):
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
    
    # Construir histórico para o prompt
    history_text = "\n".join([f"Usuário: {msg['content']}" if msg['role'] == 'user' else f"Assistente: {msg['content']}" for msg in history])

    if context.strip():  # Se houver contexto relevante nos PDFs
        prompt = f"Você é um especialista em informática e hardware simpático e está respondendo a pergunta de um usuário. O usuário pergunta: {question}. Responda a pergunta do usuário de forma amigável e informativa utilizando o contexto: {context}. **Caso não saiba a resposta, diga que não sabe.** e de respostar que sejam completas, porem não grandes. use emojis apenas no final de todas as respostas.\n\nHistórico:\n{history_text}\n\nNova pergunta: {question}. lembrando Você é um especialista em informática e hardware simpático ou seja você nao sabe coisas alem da sua especialide. porem se for fora da sua especilaide, mas tiver a infomação na sua base dados fornecidos, responda o usuario. mesmo se uma nova conversa for iniciada."
    else:
        prompt = f"Você é um assistente especializado em informática e hardware. Responda a pergunta do usuário da melhor forma possível, porem não de respostar grandes. Pergunta: {question}\n\nHistórico:\n{history_text}\n\nNova pergunta: {question}. lembrando Você é um especialista em informática e hardware simpático ou seja você nao sabe coisas alem da sua especialide. mesmo se uma nova conversa for iniciada. porem se for fora da sua especilaide, mas tiver a infomação na sua base dados fornecidos, responda o usuario."
    
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
if "conversations" not in st.session_state:
    st.session_state["conversations"] = [{"title": "Novo Chat", "messages": [{"role": "assistant", "content": "Olá! Como posso ajudar você hoje?"}]}]
if "delete_confirm" not in st.session_state:
    st.session_state["delete_confirm"] = None
if "current_conversation_index" not in st.session_state:
    st.session_state["current_conversation_index"] = 0

# Adicionar uma nova conversa
def add_new_conversation():
    new_title = f"Novo Chat {len(st.session_state['conversations']) + 1}"
    st.session_state["conversations"].append({"title": new_title, "messages": [{"role": "assistant", "content": "Olá! Como posso ajudar você hoje?"}]})
    st.session_state["current_conversation_index"] = len(st.session_state['conversations']) - 1

# Sidebar para gerenciar conversas
st.sidebar.title("Conversas")
if st.sidebar.button("Nova Conversa"):
    add_new_conversation()

# Selecionar conversa existente
conversation_titles = [conv["title"] for conv in st.session_state["conversations"]]
if conversation_titles:
    selected_conversation = st.sidebar.selectbox("Selecionar Conversa", conversation_titles, index=st.session_state["current_conversation_index"])
    st.session_state["current_conversation_index"] = conversation_titles.index(selected_conversation)
else:
    st.sidebar.write("Nenhuma conversa disponível.")

# Função para editar o título da conversa
def edit_conversation_title(index, new_title):
    st.session_state["conversations"][index]["title"] = new_title

# Função para apagar uma conversa
def delete_conversation(index):
    if st.session_state["delete_confirm"] == index:
        st.session_state["conversations"].pop(index)
        st.session_state["delete_confirm"] = None
        st.session_state["current_conversation_index"] = max(0, len(st.session_state["conversations"]) - 1)
    else:
        st.session_state["delete_confirm"] = index

# Exibir conversas existentes
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

# Função para validar a entrada do usuário
def is_valid_input(prompt):
    if re.search(r'[a-zA-Z0-9]', prompt) and len(prompt) > 2:
        return True
    return False

# Selecionar a conversa atual
if st.session_state["conversations"]:
    current_conv = st.session_state["conversations"][st.session_state["current_conversation_index"]]
else:
    current_conv = None

# Exibir mensagens na interface do chat
if current_conv:
    for message in current_conv["messages"]:
        st.chat_message(message["role"]).write(message["content"])
else:
    st.warning("Nenhuma conversa disponível. Por favor, inicie uma nova conversa na barra lateral.")

# Entrada do usuário
if question := st.chat_input("Qual é a sua dúvida hoje?"):
    question = question.strip().lower()
    if question and is_valid_input(question):
        if current_conv is not None:
            history = current_conv["messages"]
            current_conv["messages"].append({"role": "user", "content": question})
            
            # Mostrar mensagem do usuário imediatamente
            st.chat_message("user").write(question)
            
            with st.spinner("Processando..."):
                try:
                    # Chamar a função que interage com o modelo de IA utilizando o histórico
                    response = chatbot_interaction(question, history)
                    current_conv["messages"].append({"role": "assistant", "content": response})
                    st.chat_message("assistant").write(response)
                    st.experimental_rerun()  # Forçar a atualização da interface
                except Exception as e:
                    print(f"Erro ao chamar API: {e}")
                    current_conv["messages"].append({"role": "assistant", "content": f"Desculpe, não consegui entender sua pergunta. Erro: {e}"})
                    st.chat_message("assistant").write("Desculpe, não consegui entender sua pergunta.")
        else:
            st.warning("Por favor, selecione uma conversa ou inicie uma nova.")
    else:
        st.warning("Por favor, insira uma pergunta válida.")
