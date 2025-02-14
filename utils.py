from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
# from app2 import upload_file




chunk_size = 500
chunk_overlap = 50


def load_split_pdf_file(file):
    loader = PyMuPDFLoader(file)
    pages = loader.load()
    # pdf_page_content = [page.page_content for page in pages]

    # Initialize the RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function=len,
        #separators= ["\n\n", "\n", " ", ""]
    )

    # Split the text into smaller chunks
    chunks = text_splitter.split_documents(pages)


    return chunks 

    

def QA_Chain_Retrieval(query,vectordb ):
    try:
        # Formatting function for documents
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Prompt template string
        prompt_str = """
        Answer the user question based only on the following context:
        {context}

        Question: {question}
        """
        
        # Create a chat prompt template
        _prompt = ChatPromptTemplate.from_template(prompt_str)
        
        # Set the number of chunks to retrieve
        num_chunks = 3
        
        # Set up the retriever
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": num_chunks}
        )
        
        # Set up the chain components
        chat_llm = ChatOpenAI(model_name="gpt-4o-mini")
        query_fetcher = itemgetter("question")
        setup = {
            "question": query_fetcher,
            "context": query_fetcher | retriever | format_docs
        }
        
        # Define the final chain
        _chain = setup | _prompt | chat_llm
        
        # Execute the chain and fetch the response
        response = _chain.invoke({"question": query})
        return response
    
    except Exception as e:
        return f"Error executing retrieval chain: {str(e)}"