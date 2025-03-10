from typing import Annotated, List
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import uuid
import streamlit as st
import os
import time

# Crear directorio de archivos si no existe
os.makedirs("files", exist_ok=True)
# Definir el estado del chat
class ChatState2(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    thread_id: str
    custom_checkpoint_id: str

 # Función para preguntas simples
def simple_question(state: ChatState2):
    # Lógica para responder preguntas simples
    # ... (similar a call_model)
    return {"messages": [AIMessage(content=response)], "thread_id": thread_id, "custom_checkpoint_id": checkpoint_id}

# Función para preguntas complejas
def complex_question(state: ChatState2):
    # Lógica para responder preguntas complejas (múltiples pasos)
    # ... (puede incluir llamadas a otras funciones o herramientas)
    return {"messages": [AIMessage(content=response)], "thread_id": thread_id, "custom_checkpoint_id": checkpoint_id}

# Función para determinar la complejidad de la pregunta
def determine_complexity(state: ChatState2):
    messages = state["messages"]
    question = messages[-1].content

    # Lógica para determinar si la pregunta es simple o compleja
    if "información detallada" in question.lower() or "múltiples pasos" in question.lower():
        return "complex"
    else:
        return "simple"

# ... (resto del código)

# Función para cargar y procesar un PDF
@st.cache_resource
def procesar_pdf(file_path):
    """Carga el PDF, lo divide en fragmentos y crea el vectorstore."""
    loader = PyPDFLoader(file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    all_splits = text_splitter.split_documents(data)
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


# Función para llamar al modelo con retrieval (estrategias de recuperación de documentos)
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

# Configurar el gráfico de LangGraph
graph_builder = StateGraph(ChatState2)
graph_builder.add_node("determine_complexity", determine_complexity)
graph_builder.add_node("simple", simple_question)
graph_builder.add_node("complex", complex_question)

graph_builder.set_entry_point("determine_complexity")

graph_builder.add_conditional_edges(
    "determine_complexity",
    {
        "simple": "simple",  # Mantenemos los nombres como strings
        "complex": "complex", # Mantenemos los nombres como strings
    },
)

graph_builder.add_edge("simple", END)
graph_builder.add_edge("complex", END)

# Configurar la memoria
memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# Configuración de Streamlit
st.title("Asistente de Consulta para PDFs 📄")
uploaded_file = st.file_uploader("Sube tu PDF", type='pdf')

# Guardar el Archivo Subido
if uploaded_file is not None:
    file_path = f"files/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.session_state.vectorstore = procesar_pdf(file_path)

# Inicializar el estado del chat
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {"messages": [], "thread_id": str(uuid.uuid4()), "custom_checkpoint_id": str(uuid.uuid4())}

# Manejar la interacción del usuario con Streamlit
if user_input := st.chat_input("Haz una pregunta:", key="user_input"):
    user_message = HumanMessage(content=user_input)
    st.session_state.chat_history["messages"].append(user_message)

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("El asistente está respondiendo..."):
            response = graph.invoke(st.session_state.chat_history, config={"thread_id": st.session_state.chat_history["thread_id"], "custom_checkpoint_id": st.session_state.chat_history["custom_checkpoint_id"]})
            full_response = response["messages"][-1].content
            message_placeholder = st.empty()
            for chunk in full_response.split():
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            st.session_state.chat_history["messages"].append(AIMessage(content=full_response))