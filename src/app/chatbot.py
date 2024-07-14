from search import similarity_search
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def calculate_embeddings(texts):
    embeddings = model.encode(texts)
    return embeddings

def chatbot_interaction(question, history, documents, embeddings, llm):
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

    # Calcula os embeddings da pergunta e dos documentos
    question_embedding = calculate_embeddings([question])[0]

    # Encontra os documentos mais similares
    top_k = 5  # número de documentos mais similares a serem recuperados
    similar_docs_indices = similarity_search(question_embedding, embeddings, k=top_k)

    # Constrói o contexto com os documentos mais similares encontrados
    context = " ".join([documents[idx] for idx in similar_docs_indices])

    if len(history) <= 1:
        prompt = f"Você é um assistente especializado em informática e hardware. Responda a pergunta do usuário da melhor forma possível, mas não dê respostas longas. Pergunta: {question}"
    else:
        history_text = "\n".join([f"Usuário: {msg['content']}" if msg['role'] == 'user' else f"Assistente: {msg['content']}" for msg in history])
        prompt = f"Você é um especialista em informática e hardware e programação e está respondendo a pergunta de um usuário e voce responde perguntas apenas com base nos. O usuário pergunta: {question}. Responda a pergunta do usuário de forma profissional e informativa utilizando o contexto: {context}. **Caso não saiba a resposta, diga que não sabe.** \n\nHistórico:\n{history_text}\n\nNova pergunta: {question}. Lembre-se que você é um especialista em informática e hardware e programação  e programação simpático e profissional **voce nao usa emoji**, ou seja, você não sabe coisas além da sua especialidade. **Porém se a informação estiver na sua base de dados fornecida responda ao usuário mesmo que não seja sua especialidade. e nao fale que voce tem em sua base de dados jamais, e sobre pessoas, voce saberá responder das pessoas que estao no dados fornecidos**\n\n"

    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Desculpe, não consegui entender sua pergunta. Erro: {e}"
