import streamlit as st
import os

# Importaciones de motor de IA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. CONFIGURACIÓN DE PÁGINA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")
LOGO_URL_RAW = "https://github.com/rocatmo1975/asistente-iedtaca/blob/main/logo.png?raw=true"
NOMBRE_APP = "ASISTENTE IA IEDTACA"

st.set_page_config(
    page_title=NOMBRE_APP,
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🏫",
    layout="wide"
)

# ESTILOS PERSONALIZADOS
st.markdown(f"""
    <head>
        <link rel="icon" type="image/png" href="{LOGO_URL_RAW}">
        <link rel="apple-touch-icon" href="{LOGO_URL_RAW}">
    </head>
    <style>
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        .stChatMessage {{ border-radius: 15px; border: 1px solid #f0f2f6; }}
        .stChatInput {{ border-radius: 10px; }}
    </style>
""", unsafe_allow_html=True)

# --- 2. DISEÑO DE INTERFAZ ---
st.markdown(f"<h1 style='text-align: center; color: #1E3A8A;'>{NOMBRE_APP}</h1>", unsafe_allow_html=True)

if os.path.exists(LOGO_PATH):
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        try:
            st.image(LOGO_PATH, use_container_width=True)
        except:
            st.write("🏫")
st.markdown("<p style='text-align: center; color: gray;'>Sistema de Inteligencia Institucional - Carmen de Ariguaní</p>", unsafe_allow_html=True)
st.markdown("---")

# --- 3. LÓGICA DE IA DE ALTA CAPACIDAD (MMR + SOURCE TRACKING) ---
api_key = st.secrets.get("OPENAI_API_KEY")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

@st.cache_resource(show_spinner="Procesando múltiples documentos... Esto optimiza la precisión.")
def inicializar_ia(folder_path, _api_key):
    if not _api_key: return None
    os.environ["OPENAI_API_KEY"] = _api_key
    
    if not os.path.exists(folder_path): return None
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    if not pdf_files: return None
        
    try:
        documentos_completos = []
        for pdf in pdf_files:
            loader = PyPDFLoader(os.path.join(folder_path, pdf))
            docs = loader.load()
            # SELLO DE FUENTE: Guardamos el nombre del archivo en cada página
            for doc in docs:
                doc.metadata["source"] = pdf
            documentos_completos.extend(docs)
        
        # Fragmentos quirúrgicos (450 caracteres) para navegar entre mucho texto
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=450, 
            chunk_overlap=80,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        textos_fragmentados = text_splitter.split_documents(documentos_completos)
        vector_db = FAISS.from_documents(textos_fragmentados, OpenAIEmbeddings())
        
        # BUSCADOR DE DIVERSIDAD (MMR): Analiza 30 opciones para darte las 12 más variadas
        retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 12, "fetch_k": 30, "lambda_mult": 0.3}
        )
        
        model = ChatOpenAI(model="gpt-4o", temperature=0)
        
        template = """
        Eres el Asistente Experto de la IEDTACA. Tu base de conocimiento es extensa.
        
        CONTEXTO RECUPERADO DE DOCUMENTOS:
        {context}

        PREGUNTA:
        {question}

        REGLAS DE RESPUESTA:
        1. Identifica nombres, fechas y resoluciones específicas. 
        2. Si hay varios documentos que mencionan lo mismo, prioriza la información del archivo con el nombre más actual o reciente.
        3. Indica de qué documento extrajiste la información clave.
        4. Si no encuentras el dato exacto, menciona qué temas relacionados sí aparecen.

        RESPUESTA INSTITUCIONAL:
        """
        prompt = ChatPromptTemplate.from_template(template)

        # Función para formatear incluyendo la fuente del archivo
        def format_docs(docs):
            return "\n\n".join([f"ARCHIVO: {d.metadata['source']}\nCONTENIDO: {d.page_content}" for d in docs])

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt | model | StrOutputParser()
        )
        return chain
    except Exception as e:
        st.error(f"Error al indexar documentos: {e}")
        return None

# --- 4. CHAT INTERACTIVO ---
if not api_key:
    st.error("❌ Configura la OPENAI_API_KEY en Streamlit Secrets.")
else:
    rag_chain = inicializar_ia(DOCS_DIR, api_key)
    
    if rag_chain:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt_input := st.chat_input("Consulta manuales, circulares o reglamentos..."):
            st.session_state.messages.append({"role": "user", "content": prompt_input})
            with st.chat_message("user"):
                st.markdown(prompt_input)

            with st.chat_message("assistant"):
                with st.spinner("Escaneando base de datos completa..."):
                    try:
                        respuesta = rag_chain.invoke(prompt_input)
                        st.markdown(respuesta)
                        st.session_state.messages.append({"role": "assistant", "content": respuesta})
                    except Exception as e:
                        st.error(f"Error en consulta: {e}")
    else:
        st.warning("⚠️ Sin documentos. Sube tus PDFs a la carpeta 'docs' en GitHub.")
                   
