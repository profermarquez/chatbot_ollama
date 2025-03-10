# Instalar y activar un entorno virtual
/virtualenv env         /env/Scripts/activate.bat

# requerimientos
pip install langchain langchain-community langchain-ollama langgraph streamlit pypdf typing-extensions

# para imprimir el grafo
# https://graphviz.org/download/
pip install graphviz

Ollama: Debes tener Ollama instalado y funcionando localmente, con el modelo mistral:latest descargado.

# ejecutar los archivos que utilizan streamlit
streamlit run mistral_ejemplo.py



# otros repositorios de poryectos de LLMs
https://github.com/liyichen-cly/PoG
https://github.com/SuperMedIntel/Medical-Graph-RAG
https://github.com/glee4810/ehrsql-2024
https://github.com/bflashcp3f/synthetic-clinical-qa
