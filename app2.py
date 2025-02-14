from fastapi import FastAPI, File, UploadFile
from utils import load_split_pdf_file, QA_Chain_Retrieval
import tempfile
import os
from langchain_openai import OpenAIEmbeddings
import uvicorn
from Pinecone_class import pincoine_ob
from dotenv import load_dotenv
load_dotenv()

pincone_api_key = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # Correct embedding model
pincoine_ob.create_index(index_name="all-data", dimentions=1536)
# Check if the index is available
pincoine_ob.check_index(index="all-data")

app = FastAPI()

@app.post("/upload_file/")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    file_extension = file.filename.split(".")[-1].lower()

    fd, tmp_file_path = tempfile.mkstemp(suffix=f".{file_extension}")

    with os.fdopen(fd, 'wb') as tmp_file:
        tmp_file.write(contents)

    if file_extension == "pdf":
        chunks = load_split_pdf_file(tmp_file_path)
        if not chunks:
            return {"error": "Failed to process the PDF"}

        pincoine_ob.insert_data_in_namespace(chunks, embeddings=embeddings, index_name="all-data", name_space="pdf-data")
        return {"message": "File processed and stored successfully"}

    else:
        raise TypeError("Not supported file format")


vectordb = pincoine_ob.retrieve_from_namespace(index_name="all-data", embeddings=embeddings, name_space="pdf-data")

@app.post("/retrieve")
async def retrieve(query: str):
    query = query
    # Retrieve results from Qdrant based on the query
    results = QA_Chain_Retrieval(query=query,vectordb=vectordb)
    print(results)
    return results.content



if __name__ == "__main__":
    uvicorn.run("app2:app", host="127.0.0.1", port=8000, reload=True)