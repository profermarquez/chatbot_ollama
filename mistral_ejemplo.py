from typing import Annotated, List
import uuid
import os
import time

import streamlit as st

from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from typing_extensions import TypedDict

# Crear directorio de archivos si no existe
os.makedirs("files", exist_ok=True)

# === CONFIGURACIÓN DE MODELOS ===
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral:latest"


# === FUNCIONES ===
@st.cache_resource
def procesar_pdf_en_partes(file_path):
    """Carga el PDF por partes, muestra progreso y crea el vectorstore sin colgar Streamlit."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)

    vectorstore = Chroma(
        embedding_function=OllamaEmbeddings(model=EMBED_MODEL),
        persist_directory="vectorstore"
    )

    total_pages = len(pages)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, page in enumerate(pages):
        splits = text_splitter.split_documents([page])
        vectorstore.add_documents(splits)

        # Actualizar visualmente el progreso cada 10 páginas
        if i % 10 == 0 or i == total_pages - 1:
            percent_complete = (i + 1) / total_pages
            progress_bar.progress(percent_complete)
            status_text.write(f"📄 Procesadas {i + 1} de {total_pages} páginas...")

    vectorstore.persist()
    status_text.success("✅ PDF procesado correctamente.")
    return vectorstore


# === CONFIGURAR EL MODELO DE LENGUAJE ===
llm = OllamaLLM(
    base_url="http://localhost:11434",
    model=LLM_MODEL,
    verbose=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# === PROMPT ===
template = """
Eres un asistente experto en responder preguntas basándote en el contenido proporcionado. Responde siempre en español de manera clara y precisa.

📌 **Contexto relevante:** {context}
📌 **Historial de la conversación:** {history}

👤 **Usuario:** {question}
🤖 **Asistente:**
"""

# === ESTADO DEL CHAT ===
class ChatState2(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    thread_id: str
    custom_checkpoint_id: str


# === FUNCIÓN PRINCIPAL DE RESPUESTA ===
def call_model(state: ChatState2):
    messages = state["messages"]
    thread_id = state["thread_id"]
    checkpoint_id = state["custom_checkpoint_id"]

    # Recuperar documentos relevantes del vectorstore
    if 'vectorstore' in st.session_state:
        retriever = st.session_state.vectorstore.as_retriever()
        retrieved_docs = retriever.get_relevant_documents(messages[-1].content)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
    else:
        context = ""

    # Crear el prompt con el contexto y el historial
    conversation_history = "\n".join([msg.content for msg in messages])
    prompt_value = template.format(history=conversation_history, context=context, question=messages[-1].content)

    # Invocar el LLM con el prompt
    response = llm.invoke(prompt_value)
    return {"messages": [AIMessage(content=response)], "thread_id": thread_id, "custom_checkpoint_id": checkpoint_id}


# === CONFIGURAR LANGGRAPH ===
graph_builder = StateGraph(ChatState2)
graph_builder.add_node("model", call_model)
graph_builder.set_entry_point("model")
graph_builder.add_edge("model", END)

memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)


# === INTERFAZ STREAMLIT ===
st.title("Asistente de Consulta para PDFs 📄")
uploaded_file = st.file_uploader("Sube tu PDF", type='pdf')

if uploaded_file is not None:
    file_path = f"files/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.session_state.vectorstore = procesar_pdf_en_partes(file_path)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {"messages": [], "thread_id": str(uuid.uuid4()), "custom_checkpoint_id": str(uuid.uuid4())}

if user_input := st.chat_input("Haz una pregunta:", key="user_input"):
    user_message = HumanMessage(content=user_input)
    st.session_state.chat_history["messages"].append(user_message)

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("El asistente está respondiendo..."):
            response = graph.invoke(st.session_state.chat_history, config={
                "thread_id": st.session_state.chat_history["thread_id"],
                "custom_checkpoint_id": st.session_state.chat_history["custom_checkpoint_id"]
            })
            full_response = response["messages"][-1].content
            message_placeholder = st.empty()
            for chunk in full_response.split():
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            st.session_state.chat_history["messages"].append(AIMessage(content=full_response))
