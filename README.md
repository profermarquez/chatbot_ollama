# Instalar y activar un entorno virtual con virtualenv
/virtualenv env         /env/Scripts/activate.bat

# requerimientos
pip install langchain langchain-community langchain-ollama langgraph streamlit pypdf typing-extensions chromadb

# para imprimir el grafo
# https://graphviz.org/download/
pip install graphviz

Ollama: Debes tener Ollama instalado y funcionando localmente, con el modelo mistral:latest descargado.

# Limpiar base de datos vectorial
rm -rf vectorstore/   

# ejecutar los archivos que utilizan streamlit
streamlit run mistral_ejemplo.py



# otros repositorios de poryectos de LLMs
https://github.com/liyichen-cly/PoG
https://github.com/SuperMedIntel/Medical-Graph-RAG
https://github.com/glee4810/ehrsql-2024
https://github.com/bflashcp3f/synthetic-clinical-qa
