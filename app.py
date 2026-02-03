import streamlit as st
import os

# Importaciones de motor de IA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. CONFIGURACIÓN DE PÁGINA E ICONO ---
# Definimos las rutas primero
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")
NOMBRE_APP = "ASISTENTE IA IEDTACA"

# Aquí configuramos que el logo sea el icono de la pestaña y de la app instalada
st.set_page_config(
    page_title=NOMBRE_APP,
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🏫",
    layout="wide"
)

# --- 2. DISEÑO DE INTERFAZ ---
st.markdown(f"<h1 style='text-align: center;'>{NOMBRE_APP}</h1>", unsafe_allow_html=True)

# Mostrar el escudo en el cuerpo de la página
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

# --- 3. LÓGICA DE IA CON CACHÉ ---
api_key = st.secrets.get("OPENAI_API_KEY")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

@st.cache_resource(show_spinner="Cargando base de conocimiento... Esto solo tarda la primera vez.")
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
        paginas = []
        for pdf in pdf_files:
            ruta_pdf = os.path.join(folder_path, pdf)
            loader = PyPDFLoader(ruta_pdf)
            paginas.extend(loader.load())
        
        vector_db = FAISS.from_documents(paginas, OpenAIEmbeddings())
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        
        model = ChatOpenAI(model="gpt-4o", temperature=0)
        
        template = """
        Eres el ASISTENTE IA IEDTACA. Responde de forma amable y profesional basándote en el contexto.
        Si no sabes la respuesta basándote en el contexto, dilo amablemente.
        
        Contexto: {context}
        Pregunta: {question}
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

if not api_key:
    st.error("❌ Falta la API KEY en los Secrets.")
else:
    rag_chain = inicializar_ia(DOCS_DIR, api_key)
    
    if rag_chain:
        st.success("✅ Asistente activo.")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt_input := st.chat_input("Escribe tu pregunta aquí..."):
            st.session_state.messages.append({"role": "user", "content": prompt_input})
            with st.chat_message("user"):
                st.markdown(prompt_input)

            with st.chat_message("assistant"):
                with st.spinner("Buscando en los documentos..."):
                    try:
                        respuesta = rag_chain.invoke(prompt_input)
                        st.markdown(respuesta)
                        st.session_state.messages.append({"role": "assistant", "content": respuesta})
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.warning("No hay PDFs en la carpeta 'docs'.")
