# este ejemplo de chat guarda un modelo en una base de datos vectorial y delira un poco mas en la respuesta
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader  
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st
import os
import time

# Crear directorio de archivos si no existe
os.makedirs("files", exist_ok=True)

# Función para cargar y procesar un PDF
@st.cache_resource
def procesar_pdf(file_path):
    """Carga el PDF, lo divide en fragmentos y crea el vectorstore."""
    loader = PyPDFLoader(file_path)
    data = loader.load()

    # Dividir el texto en fragmentos
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    all_splits = text_splitter.split_documents(data)

    # Crear y persistir el vectorstore
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=OllamaEmbeddings(model="mistral"),
        persist_directory="vectorstore"
    )
    vectorstore.persist()
    return vectorstore

# Configurar el modelo de lenguaje
llm = OllamaLLM(
    base_url="http://localhost:11434",
    model="mistral:latest",
    verbose=True,
    callbacks=[StreamingStdOutCallbackHandler()]  
)

# Definir la plantilla de prompt en español
template = """
Eres un asistente experto en responder preguntas basándote en el contenido proporcionado. Responde siempre en español de manera clara y precisa.

📌 **Contexto relevante:** {context}
📌 **Historial de la conversación:** {history}

👤 **Usuario:** {question}
🤖 **Asistente:**
"""

prompt = PromptTemplate(
    input_variables=["history", "context", "question"],  # ✅ Se usa 'question' en lugar de 'query'
    template=template,
)

# Configurar la memoria del chat
memory = ConversationBufferMemory(memory_key="history", return_messages=True, input_key="question")  # ✅ Se usa 'question'

# Configuración de Streamlit
st.title("Asistente de Consulta para PDFs 📄")

uploaded_file = st.file_uploader("Sube tu PDF", type='pdf')

# Inicializar el historial del chat
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Procesar el archivo subido
if uploaded_file is not None:
    file_path = f"files/{uploaded_file.name}"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Cargar el PDF y generar el vectorstore
    st.session_state.vectorstore = procesar_pdf(file_path)

# Inicializar la cadena de preguntas y respuestas en el estado de sesión
if 'qa_chain' not in st.session_state and 'vectorstore' in st.session_state:
    retriever = st.session_state.vectorstore.as_retriever()

    st.session_state.qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        verbose=True,
        return_source_documents=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt,
            "memory": memory,
        }
    )

# Manejar la interacción del usuario con Streamlit
if user_input := st.chat_input("Haz una pregunta:", key="user_input"):
    user_message = {"role": "user", "message": user_input}
    st.session_state.chat_history.append(user_message)

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("El asistente está respondiendo..."):
            response = st.session_state.qa_chain.invoke({"query": user_input})  # ✅ Se usa 'question' en lugar de 'query'

        message_placeholder = st.empty()
        full_response = ""

        # Si hay documentos relevantes en el PDF, úsalos primero
        if "source_documents" in response and response["source_documents"]:
            pdf_context = "\n".join([doc.page_content for doc in response["source_documents"][:3]])
            full_response += f"📖 **Información del documento:**\n\n{pdf_context}\n\n"

        # Añadir la respuesta generada por el modelo LLM
        full_response += response.get('result', 'No se encontró una respuesta relevante.')

        # Simular escritura en tiempo real
        for chunk in full_response.split():
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    chatbot_message = {"role": "assistant", "message": full_response}
    st.session_state.chat_history.append(chatbot_message)
