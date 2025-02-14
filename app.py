from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from Pinecone_class import pincoine_ob
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from fastapi import FastAPI, File, UploadFile
import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

chunk_size = 500
chunk_overlap = 50

def load_split_pdf_file(file_path):
    try:
        loader = PyMuPDFLoader(file_path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        data = text_splitter.split_documents(pages)
        return data

    except Exception as e:
        print(f"Error loading and splitting PDF file: {str(e)}")
        

pincone_api_key = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
pincoine_ob.create_index(index_name="storage", dimentions = 1536)
pincoine_ob.check_index(index="storage")

app = FastAPI()

@app.post("/upload_file/")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    file_extension = file.filename.split(".")[-1].lower()


    if file_extension == "pdf":
        chunks = load_split_pdf_file(contents)
        if not chunks:
            return {"error": "Failed to process the PDF"}

        pincoine_ob.insert_data_in_namespace(chunks, embeddings=embeddings, index_name="storage", name_space="pdf-data")
        return {"message": "File processed and stored successfully"}

    else:
        raise TypeError("Not supported file format")


# vectordb = pincoine_ob.retrieve_from_namespace(index_name="storage", embeddings=embeddings, name_space="pdf-data")

def QA_Chain_Retrieval(query, vectordb):
    try:
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        prompt_str = """
        Answer the user question based only on the following context:
        {context}

        Question: {question}
        """

        _prompt = ChatPromptTemplate.from_template(prompt_str)

        num_chunks = 3
        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": num_chunks})

        chat_llm = ChatOpenAI(model_name="gpt-4o-mini")

        retrieved_docs = retriever.invoke(query)
        formatted_context = format_docs(retrieved_docs)

        response = chat_llm.invoke(_prompt.format(context=formatted_context, question=query))
        return response.content

    except Exception as e:
        return f"Error executing retrieval chain: {str(e)}"
@app.post("/retrieve")
async def retrieve(query: str):
    vectordb = pincoine_ob.retrieve_from_namespace(index_name="storage", embeddings=embeddings, name_space="pdf-data")
    
    if not vectordb:
        return {"error": "No data found in vector database"}

    response = QA_Chain_Retrieval(query, vectordb)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8100, reload=True)
