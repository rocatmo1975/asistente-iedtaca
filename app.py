import streamlit as st
import os

# Importaciones de motor de IA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. CONFIGURACIÓN DE IDENTIDAD ---
NOMBRE_APP = "ASISTENTE IA IEDTACA"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")

st.set_page_config(page_title=NOMBRE_APP, page_icon="🏫", layout="wide")

# --- 2. ESTILO Y POSICIONAMIENTO DEL ESCUDO ---
# Usamos columnas para centrar el logo
col_espacio_izq, col_logo, col_espacio_der = st.columns([2, 1, 2])

with col_logo:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    else:
        st.warning("⚠️ logo.png no encontrado")

# Título centrado con HTML
st.markdown(f"<h1 style='text-align: center;'>{NOMBRE_APP}</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Sistema de consulta técnica - Institución Educativa Departamental Técnica Agropecuaria Carmen de Ariguaní</p>", unsafe_allow_html=True)

# --- 3. BARRA LATERAL ---
st.sidebar.header("Seguridad")
api_key = st.sidebar.text_input("Ingresa tu OpenAI API Key", type="password")

# --- 4. LÓGICA DE INTELIGENCIA ARTIFICIAL ---
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    st.markdown("---")
    
    archivo_pdf = st.file_uploader("📂 Cargar documentos (Manual de Convivencia, PEI, etc.)", type="pdf")
    
    if archivo_pdf:
        temp_path = os.path.join(BASE_DIR, "temp_doc.pdf")
        with open(temp_path, "wb") as f:
            f.write(archivo_pdf.getbuffer())
        
        with st.spinner("Procesando archivos de la institución..."):
            try:
                loader = PyPDFLoader(temp_path)
                paginas = loader.load()
                
                vector_db = FAISS.from_documents(paginas, OpenAIEmbeddings())
                retriever = vector_db.as_retriever()
                
                model = ChatOpenAI(model="gpt-4o", temperature=0)
                
                template = """
                Eres el ASISTENTE IA IEDTACA. 
                Tu misión es ayudar a los docentes respondiendo de forma clara.
                Usa solo este contexto institucional: {context}
                
                Pregunta: {question}
                """
                prompt = ChatPromptTemplate.from_template(template)

                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt | model | StrOutputParser()
                )

                st.success("✅ Conocimiento cargado correctamente.")
                
                # Área de chat
                pregunta = st.text_input("💬 ¿Qué consulta deseas realizar?")
                if pregunta:
                    with st.spinner("Analizando documentos oficiales..."):
                        respuesta = rag_chain.invoke(pregunta)
                        st.markdown("### Respuesta del Asistente:")
                        st.info(respuesta)
            
            except Exception as e:
                st.error(f"Error técnico: {e}")
else:
    st.info("👈 Ingresa la clave API en la barra lateral para comenzar.")