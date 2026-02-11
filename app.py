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

# --- 1. CONFIGURACIÓN DE PÁGINA E ICONO ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")
LOGO_URL_RAW = "https://github.com/rocatmo1975/asistente-iedtaca/blob/main/logo.png?raw=true"
NOMBRE_APP = "ASISTENTE IA IEDTACA"

st.set_page_config(
    page_title=NOMBRE_APP,
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🏫",
    layout="wide"
)

# INYECCIÓN DE CÓDIGO HTML PARA ICONO Y ESTILO
st.markdown(f"""
    <head>
        <link rel="icon" type="image/png" href="{LOGO_URL_RAW}">
        <link rel="apple-touch-icon" href="{LOGO_URL_RAW}">
        <meta name="mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-capable" content="yes">
    </head>
    <style>
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        .stChatMessage {{ border-radius: 15px; }}
    </style>
""", unsafe_allow_html=True)

# --- 2. DISEÑO DE INTERFAZ ---
st.markdown(f"<h1 style='text-align: center;'>{NOMBRE_APP}</h1>", unsafe_allow_html=True)

if os.path.exists(LOGO_PATH):
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        try:
            st.image(LOGO_PATH, use_container_width=True)
        except:
            st.write("🏫")
else:
    st.markdown("<h3 style='text-align: center;'>🏫</h3>", unsafe_allow_html=True)

st.markdown("<p style='text-align: center; color: gray;'>Sistema de consulta técnica - Carmen de Ariguaní</p>", unsafe_allow_html=True)
st.markdown("---")

# --- 3. LÓGICA DE IA OPTIMIZADA (RAG DE ALTA PRECISIÓN) ---
api_key = st.secrets.get("OPENAI_API_KEY")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

@st.cache_resource(show_spinner="Sincronizando base de conocimiento... Esto garantiza respuestas precisas.")
def inicializar_ia(folder_path, _api_key):
    if not _api_key:
        return None
    
    os.environ["OPENAI_API_KEY"] = _api_key
    
    if not os.path.exists(folder_path):
        return None
    
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    if not pdf_files:
        return None
        
    try:
        documentos_completos = []
        for pdf in pdf_files:
            ruta_pdf = os.path.join(folder_path, pdf)
            loader = PyPDFLoader(ruta_pdf)
            documentos_completos.extend(loader.load())
        
        # AJUSTE DE PRECISIÓN: Fragmentos más pequeños (600) para evitar omisiones
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600, 
            chunk_overlap=120,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        textos_fragmentados = text_splitter.split_documents(documentos_completos)
        
        vector_db = FAISS.from_documents(textos_fragmentados, OpenAIEmbeddings())
        
        # BUSCADOR ENFOCADO: Recupera los 6 fragmentos más relevantes
        retriever = vector_db.as_retriever(search_kwargs={"k": 6})
        
        model = ChatOpenAI(model="gpt-4o", temperature=0)
        
        template = """
        Eres el ASISTENTE IA IEDTACA. Tu base de conocimiento son los documentos institucionales proporcionados.
        
        INSTRUCCIONES CRÍTICAS:
        1. Responde basándote estrictamente en el contexto.
        2. Si la información parece estar dispersa, relaciónala para dar una respuesta coherente.
        3. Si la respuesta está en los documentos, sé específico y cita el nombre del documento si es posible.
        4. No respondas "no sé" si hay información relacionada que pueda ayudar al docente. 
        5. Solo si no hay rastro del tema, indica que no se encuentra en la base de datos actual.

        Contexto: {context}
        Pregunta: {question}
        Respuesta Institucional:
        """
        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt | model | StrOutputParser()
        )
        return chain
    except Exception as e:
        st.error(f"Error procesando documentos: {e}")
        return None

# --- 4. EJECUCIÓN DEL CHAT ---
if not api_key:
    st.error("❌ Falta la API KEY en los Secrets de Streamlit.")
else:
    rag_chain = inicializar_ia(DOCS_DIR, api_key)
    
    if rag_chain:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt_input := st.chat_input("Realiza tu consulta sobre la normativa institucional aquí..."):
            st.session_state.messages.append({"role": "user", "content": prompt_input})
            with st.chat_message("user"):
                st.markdown(prompt_input)

            with st.chat_message("assistant"):
                with st.spinner("Analizando manuales y reglamentos..."):
                    try:
                        respuesta = rag_chain.invoke(prompt_input)
                        st.markdown(respuesta)
                        st.session_state.messages.append({"role": "assistant", "content": respuesta})
                    except Exception as e:
                        st.error(f"Hubo un problema al consultar la base de datos: {e}")
    else:
        st.warning("⚠️ No se cargó la base de conocimiento. Verifica que los PDFs estén en la carpeta 'docs'.")
