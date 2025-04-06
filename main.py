import streamlit as st

from PyPDF2 import PdfReader
from docx import Document

from langchain.text_splitter import CharacterTextSplitter
#from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.embeddings import NomicEmbedding
from langchain.vectorstores import FAISS
from langchain.llms import Ollama

from langchain.memory import ConversationBufferMemory
from langchain.schema.messages import SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import LLMChain
import requests
from langchain.embeddings.base import Embeddings

class OllamaEmbedding(Embeddings):
    def __init__(self, model_name="nomic-embed-text"):
        self.model_name = model_name

    def embed_documents(self, texts):
        return [get_ollama_embedding(text, self.model_name) for text in texts]

    def embed_query(self, text):
        return get_ollama_embedding(text, self.model_name)



def get_ollama_embedding(text: str, model="nomic-embed-text"):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": model, "prompt": text}
    )
    response.raise_for_status()
    return response.json()["embedding"]


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def extract_text(file):
    text=""
    try:
        if file.type == "application/pdf":
            reader = PdfReader(file)
            
            #text extract here
            for page in reader.pages:
                text = text + page.extract_text()
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(file)
            for paragraph in doc.paragraphs:
                text = text + paragraph.text + '\n'
        return text
    except Exception as e:
        st.error(f"Error processing file {file.name}: {e}")
        return None


def generate_summary(text):
    # Using Ollama instead of Gemini
    llm = Ollama(model="qwen2.5:32b", temperature=0)
#     prompt = f"""
# Create a concise professional summary of this resume, optimized for recruiters.
# Highlight:
#  - top 3-5 key skills and technologies.
#  - years of relevant experience and prominent job roles
#  - educational background and significant certifications
#  - 1-2 impact achievements or contributions

#  Format the summary for quick reading with bullet points if appropriate.
#  Resume text: {text[:5000]}
#     """
    prompt = f"""
Create **a single**, concise professional summary of this resume, optimized for recruiters.
Highlight:
 - top 3-5 key skills and technologies.
 - years of relevant experience and prominent job roles
 - educational background and significant certifications
 - 1-2 impact achievements or contributions

Format **the summary** for quick reading with bullet points if appropriate. **Provide just one summary, not options.**
Resume text: {text[:5000]}
"""
    return llm.invoke(prompt)

def main():
    st.set_page_config(page_title="AI Resume Analysis Chat - Coding Assessment 2")
    st.title("AI Resume Analysis Chat - Coding Assessment 2")
    
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
                return_messages=True,
                memory_key="chat_history"
            )
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "uploaded_filenames" not in st.session_state:
        st.session_state.uploaded_filenames = []
    if "summaries" not in st.session_state:
        st.session_state.summaries = {}

    with st.sidebar:
        st.header("Upload Resumes")

        uploaded_files = st.file_uploader(
            "Upload resumes(PDF/DOCX):",
            type=["pdf","docx"],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        
        # ollama_model = st.selectbox(
        #     "Select Ollama Model",
        #     ["llama2", "mistral", "gemma3:4b", "phi"],
        #     index=0
        # )
        ollama_model = "qwen2.5:32b"
    if uploaded_files:
        current_filenames = [file.name for file in uploaded_files]
        if current_filenames != st.session_state.uploaded_filenames or st.session_state.vectorstore is None:
            with st.spinner("Processing resumes --->"):
                st.session_state.uploaded_filenames = current_filenames
                all_text=[]
                st.session_state.summaries={}
                for file in uploaded_files:
                    text = extract_text(file)
                    if text:
                        all_text.append({"name":file.name,"text":text})
                        with st.spinner(f"Generating summary for {file.name}.."):
                           st.session_state.summaries[file.name]= generate_summary(text)
                if all_text:
                    with st.spinner("Creating vector database..."):
                        text_splitter = CharacterTextSplitter(
                            separator="\n",
                            chunk_size=1000,
                            chunk_overlap=200,
                            length_function=len
                        )
                        chunks=[]
                        for doc in all_text:
                            print(doc)
                            chunks.extend(text_splitter.split_text(doc["text"]))


                        # Using HuggingFace embeddings instead of Google embeddings
                        #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                        #embeddings = NomicEmbedding()
                        embeddings = OllamaEmbedding(model_name="nomic-embed-text")

                        # Debugging Point: Print chunks before embedding
                        print("Debugging: Type of chunks before embedding:", type(chunks))
                        if chunks and isinstance(chunks, list):
                            if chunks:
                                print("Debugging: Type of first element in chunks:", type(chunks[0]))
                                if not isinstance(chunks[0], str):
                                    print("Debugging: First element of chunks:", chunks[0]) # Inspect the content if not string

                        st.session_state.vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
                       
                        
                else:
                    st.warning("No text could be extracted")
                    st.session_state.vectorstore=None
        else:
            st.info("Resumes already processed. Using cache")
    
    if uploaded_files and st.session_state.vectorstore:
        st.header("Chat with resumes")
        with st.expander("View Candidate summaries"):
            for name, summary in st.session_state.summaries.items():
                st.subheader(name)
                st.write(summary)
                st.divider()

        retrieve = st.session_state.vectorstore.as_retriever()

        # prompt = ChatPromptTemplate.from_messages([
        #     SystemMessage(content="""You are a helpful AI Assistant who specializes in analyzing resumes for recruiters.
        #                   Use the context provided from resumes to answer questions about the candidates, their skills, experience and qualifications.
        #                   If the answer is not found within the context, truthfully respond that you cannot answer based on the provided resumes.
        #     """),
        #     MessagesPlaceholder(variable_name="chat_history"),
        #     HumanMessagePromptTemplate.from_template("""Context information for resumes: {context}
        #     Question: {question}""")
        # ])
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a helpful AI Assistant who specializes in analyzing resumes for recruiters.
                          Your goal is to answer questions about candidates based *only* on the information explicitly stated in the provided resumes.
                          Do not make assumptions or infer skills that are not directly mentioned.

                          Focus on identifying skills, experiences, and qualifications that are *clearly described* in the resumes.

                          If a question asks about a skill or qualification that is *not mentioned* in the resume text, you must respond with:
                          "Based on the resume, I cannot determine if the candidate has this skill/qualification."
                          or a similar phrase that clearly indicates the information is not available in the provided documents.

                          Answer truthfully and only based on the content of the resumes provided as context.  Do not use outside knowledge.
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("""Context information from resumes: {context}
            Question: {question}""")
        ])
        
        
        llm = Ollama(model=ollama_model,    temperature=0.3)
        
        # Set up RAG chain with Ollama
        def rag_chain(question):
            context_docs = retrieve.invoke(question)
            context_text = format_docs(context_docs)
            chain = LLMChain(
                llm=llm,
                prompt=prompt
            )
            return chain.invoke({
                "context": context_text,
                "question": question,
                "chat_history": st.session_state.memory.chat_memory.messages
            })

        for message in st.session_state.memory.chat_memory.messages:
            with st.chat_message("assistant" if message.type == "ai" else "user"):
                st.markdown(message.content)
        
        if prompt_text := st.chat_input("Ask about the resumes"):
            st.session_state.memory.chat_memory.add_user_message(prompt_text)
            with st.chat_message("user"):
                st.markdown(prompt_text)
            with st.chat_message("assistant"):
                response_container = st.empty()
                with st.spinner("Generating response..."):
                    # Debugging Point: Print prompt_text and chat_history before invoke
                    print("Debugging: Type of prompt_text:", type(prompt_text))
                    print("Debugging: Type of chat_history:", type(st.session_state.memory.chat_memory.messages))

                    # Using the RAG chain with Ollama
                    response = rag_chain(prompt_text)
                    #st.markdown(response["text"])
                    response_container.markdown(response["text"])
                    st.session_state.memory.chat_memory.add_ai_message(response["text"])
    print(get_ollama_embedding("What is the capital of France?")[:5])

if __name__=="__main__":
    main()
