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

    if question.lower() in basic_responses:
        return basic_responses[question.lower()]

    question_embedding = calculate_embeddings([question])[0]

    top_k = 21 
    similar_docs_indices = similarity_search(question_embedding, embeddings, k=top_k)

    context = " ".join([documents[idx] for idx in similar_docs_indices])
    
    history_text = "\n".join([f"Usuário: {msg['content']}" if msg['role'] == 'user' else f"Assistente: {msg['content']}" for msg in history])

    prompt = f"""
    **Você é um especialista em informática e hardware e programação super simpático e você responde perguntas com a base de dados fornecida. O usuário pergunta: {question}. Responda a pergunta do usuario de forma informativa utilizando o contexto e a base dados: {context}. **e voce nao responde perguntas de coisas que não esteja na sua base de dados ou que não seja da sua especialidade** mas se caso a pergunta não for da sua especialidade mas tiver algo relacionado na sua base de dados pode responder sem problemas mesmo que seja dados de pessoas mas apenas das pessoas que estão na sua base de dados.** evite ficar repetindo frases de saldações e olá **Caso não saiba a resposta, diga que não sabe de maneira simpatica.**

    Histórico:
    {history_text}

    Nova pergunta: {question}. **Porém se o usuario fizer uma pergunta e essa informação estiver na sua base de dados fornecida, responda ao usuário mesmo que não seja da sua especialidade**. E não fale que você tem em sua base de dados jamais de preferencia nem citar base de dados, e sobre nome de pessoas, você saberá responder das pessoas que estão na base de dados fornecidos essas informações voce podera responder tranquilamente pois é a sua função, as pessoas da base de dados são masculinas**.
    """
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Desculpe, não consegui entender sua pergunta. Erro: {e}"
