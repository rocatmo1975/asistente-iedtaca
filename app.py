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
DOCS_DIR = os.path.join(BASE_DIR, "docs")  # <--- AJUSTADO A "docs"

st.set_page_config(page_title=NOMBRE_APP, page_icon="🏫", layout="wide")

# Intentar obtener la API KEY desde los Secrets de Streamlit
api_key = st.secrets.get("OPENAI_API_KEY")

# --- 2. DISEÑO DE INTERFAZ ---
col_izq, col_logo, col_der = st.columns([2, 1, 2])
with col_logo:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)

st.markdown(f"<h1 style='text-align: center;'>{NOMBRE_APP}</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray; font-size: 18px;'>Consulta técnica: Carmen de Ariguaní</p>", unsafe_allow_html=True)

# --- 3. LÓGICA DE CARGA AUTOMÁTICA ---
if not api_key:
    st.error("❌ No se encontró la 'OPENAI_API_KEY' en los Secrets. Por favor, agrégala en Settings > Secrets.")
else:
    os.environ["OPENAI_API_KEY"] = api_key
    
    if os.path.exists(DOCS_DIR):
        pdf_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]
        
        if not pdf_files:
            st.warning(f"⚠️ No hay archivos PDF en la carpeta '{DOCS_DIR}'.")
        else:
            with st.spinner("Cargando documentos de la institución..."):
                try:
                    paginas = []
                    for pdf in pdf_files:
                        loader = PyPDFLoader(os.path.join(DOCS_DIR, pdf))
                        paginas.extend(loader.load())
                    
                    vector_db = FAISS.from_documents(paginas, OpenAIEmbeddings())
                    retriever = vector_db.as_retriever()
                    
                    model = ChatOpenAI(model="gpt-4o", temperature=0)
                    
                    template = """
                    Eres el ASISTENTE IA IEDTACA. 
                    Responde de forma clara basándote solo en el contexto institucional.
                    Si no sabes la respuesta basándote en el contexto, dilo amablemente.
                    
                    Contexto: {context}
                    Pregunta: {question}
                    """
                    prompt = ChatPromptTemplate.from_template(template)

                    def format_docs(docs):
                        return "\n\n".join(doc.page_content for doc in docs)

                    rag_chain = (
                        {"context": retriever | format_docs, "question": RunnablePassthrough()}
                        | prompt | model | StrOutputParser()
                    )

                    st.success(f"✅ {len(pdf_files)} documentos cargados correctamente.")
                    
                    # --- 4. ÁREA DE CHAT ---
                    st.markdown("---")
                    if "messages" not in st.session_state:
                        st.session_state.messages = []

                    # Mostrar historial
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])

                    # Entrada de usuario
                    if prompt_input := st.chat_input("Escribe tu duda aquí (ej. ¿Qué dice el PEI sobre la misión?)"):
                        st.session_state.messages.append({"role": "user", "content": prompt_input})
                        with st.chat_message("user"):
                            st.markdown(prompt_input)

                        with st.chat_message("assistant"):
                            with st.spinner("Analizando..."):
                                respuesta = rag_chain.invoke(prompt_input)
                                st.markdown(respuesta)
                        st.session_state.messages.append({"role": "assistant", "content": respuesta})
                
                except Exception as e:
                    st.error(f"Error al procesar: {e}")
    else:
        st.error(f"❌ No existe la carpeta '{DOCS_DIR}' en el repositorio.")

