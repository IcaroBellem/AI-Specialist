<img width="450px" src="https://github.com/user-attachments/assets/926cde2e-bd0e-4a5e-9f18-a5866bf45f24">


# TechzAI

## Descrição Geral

"TechzAI" é uma aplicação de chatbot interativa especializada em fornecer respostas sobre hardware e informática. A aplicação utiliza modelos de linguagem IA para gerar respostas precisas e informativas. O chatbot processa e indexa documentos PDF relevantes, permitindo respostas baseadas no conteúdo desses documentos. Além disso, a aplicação permite gerenciar múltiplas conversas com histórico de interações, proporcionando uma experiência personalizada ao usuário.

## Funcionalidades

1. **Divisão e Indexação de Documentos PDF**
   - Carrega documentos PDF, extrai o texto e divide-o em segmentos menores.
   - Utiliza vetorização TF-IDF para criar embeddings dos segmentos de texto, facilitando a busca por similaridade.

2. **Busca por Similaridade**
   - Implementa uma função de busca que encontra segmentos de texto relevantes nos documentos PDF com base na similaridade de cosseno entre a query do usuário e os segmentos armazenados.

3. **Interação com Chatbot**
   - Utiliza o modelo de linguagem do Google Generative AI para gerar respostas baseadas em perguntas do usuário.
   - Integra histórico de conversa para manter o contexto e fornecer respostas mais precisas.

4. **Gerenciamento de Conversas**
   - Permite iniciar, gerenciar e deletar múltiplas conversas.
   - Armazena o histórico de mensagens de cada conversa, facilitando a continuidade e a personalização das interações.

5. **Interface com Streamlit**
   - Interface web interativa construída com Streamlit.
   - Configuração da página, incluindo título, descrição e layout.
   - Funcionalidades de entrada de texto para o usuário e exibição de respostas do chatbot.

## Requisitos de Instalação

Para que a aplicação "TechzAI" funcione corretamente, é necessário instalar as seguintes bibliotecas e frameworks:

1. **Streamlit**
   - Framework para criar aplicações web interativas.
   - Instalação: `pip install streamlit`

2. **PyPDF2**
   - Biblioteca para manipulação de arquivos PDF.
   - Instalação: `pip install pypdf2`

3. **NumPy**
   - Biblioteca para operações numéricas e manipulação de arrays.
   - Instalação: `pip install numpy`

4. **joblib**
   - Biblioteca para serialização e desserialização de objetos Python.
   - Instalação: `pip install joblib`

5. **scikit-learn**
   - Biblioteca para aprendizado de máquina, incluindo vetorização de texto e cálculo de similaridade.
   - Instalação: `pip install scikit-learn`

6. **python-dotenv**
   - Biblioteca para carregar variáveis de ambiente de arquivos `.env`.
   - Instalação: `pip install python-dotenv`

7. **langchain-google-genai**
   - Biblioteca para interagir com a API do Google Generative AI.
   - Instalação: `pip install langchain-google-genai`

7. **Transformers**
   - Transformers: Biblioteca para trabalhar com modelos de linguagem pré-treinados, como BERT, GPT, etc.,facilitando o uso de NLP avançada em seus projetos.
   - Sentence Transformers: Especializada em gerar embeddings (representações vetoriais) de frases e documentos inteiros, útil para tarefas como pesquisa semântica, 
     similaridade de texto, clustering, entre outros.
   - Instalação: `pip install transformers sentence-transformers` 

## Instalação Completa

Para instalar todas as bibliotecas necessárias de uma vez, execute o seguinte comando no terminal:

```bash
pip install streamlit pypdf2 numpy joblib scikit-learn python-dotenv langchain-google-genai transformers sentence-transformers