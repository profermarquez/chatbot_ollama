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
#agregado
from typing import Dict

# Crear directorio de archivos si no existe
os.makedirs("files", exist_ok=True)

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
# Función para simular la extracción de preguntas y respuestas del PDF
def extraer_preguntas_respuestas(vectorstore) -> List[Dict[str, str]]:
    """Simula la extracción de preguntas y respuestas de los documentos en el vectorstore."""
    preguntas_respuestas = []
    for doc in vectorstore.get()['documents']: #Accedemos directamente a la lista de documentos.
        contenido = doc # El contenido esta directamente en la variable doc.
        # Simulación: Generar algunas preguntas y respuestas basadas en el contenido
        if "información importante" in contenido.lower():
            try:
                preguntas_respuestas.append({
                    "pregunta": "¿Cuál es la información importante?",
                    "respuesta": contenido.split("información importante:")[1].split(".")[0]
                })
            except IndexError:
                print("Error de indice al procesar información importante.")
        if "concepto clave" in contenido.lower():
            try:
                preguntas_respuestas.append({
                    "pregunta": "¿Qué es el concepto clave?",
                    "respuesta": contenido.split("concepto clave:")[1].split(".")[0]
                })
            except IndexError:
                print("Error de indice al procesar concepto clave.")

    return preguntas_respuestas

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

# Definir el estado del chat
class ChatState2(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    thread_id: str
    custom_checkpoint_id: str

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
graph_builder.add_node("model", call_model)
graph_builder.set_entry_point("model")
graph_builder.add_edge("model", END)

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

    # Extraer preguntas y respuestas para el ajuste fino
    preguntas_respuestas = extraer_preguntas_respuestas(st.session_state.vectorstore)
    st.session_state.preguntas_respuestas = preguntas_respuestas #Guardamos las preguntas y respuestas en la sesion de streamlit

    # Mostrar las preguntas y respuestas extraídas (para depuración)
    if preguntas_respuestas:
        st.write("Preguntas y Respuestas Extraídas para Ajuste Fino:")
        for par in preguntas_respuestas:
            st.write(f"Pregunta: {par['pregunta']}")
            st.write(f"Respuesta: {par['respuesta']}")
            st.write("---")

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