from search import similarity_search

def chatbot_interaction(question, history, documents, vectorizer, embeddings, llm):
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

    docs = similarity_search(question, documents, vectorizer, embeddings, k=5)
    context = " ".join(docs)
    
    history_text = "\n".join([f"Usuário: {msg['content']}" if msg['role'] == 'user' else f"Assistente: {msg['content']}" for msg in history])

    if context.strip():
        prompt = f"Você é um especialista em informática e hardware simpático e está respondendo a pergunta de um usuário. O usuário pergunta: {question}. Responda a pergunta do usuário de forma amigável e informativa utilizando o contexto: {context}. **Caso não saiba a resposta, diga que não sabe.** e de respostas que sejam completas, porem não grandes. use emojis apenas no final de todas as respostas.\n\nHistórico:\n{history_text}\n\nNova pergunta: {question}. lembrando Você é um especialista em informática e hardware simpático ou seja você nao sabe coisas alem da sua especialide. porem se for fora da sua especilaide, mas tiver a infomação na sua base dados fornecidos, responda o usuario. mesmo se uma nova conversa for iniciada. reforçando se estiver na sua base de dados, responda o usuario completamente com o que tem na sua base de dados."
    else:
        prompt = f"Você é um assistente especializado em informática e hardware. Responda a pergunta do usuário da melhor forma possível, porem não de respostar grandes. Pergunta: {question}\n\nHistórico:\n{history_text}\n\nNova pergunta: {question}. lembrando Você é um especialista em informática e hardware simpático ou seja você nao sabe coisas alem da sua especialide. mesmo se uma nova conversa for iniciada. porem se for fora da sua especilaide, mas tiver a infomação na sua base dados fornecidos, responda o usuario."
    
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Desculpe, não consegui entender sua pergunta. Erro: {e}"
