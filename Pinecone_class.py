from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
load_dotenv()
import os
pinecone_key = os.getenv("PINECONE_API_KEY")




class PineconeInsertRetrieval:
    def __init__(self,api_key):
        self.api_key = api_key

    # check index is exist or not 
    def check_index(self,index):
        pc = Pinecone(api_key=self.api_key)
        indexes = pc.list_indexes().names()
        if index not in indexes:
            return "Not Found index"
        elif index in indexes:
            return f"Your index name {index} Found"
    
    # create new index 
    def create_index(self,index_name,dimentions):
        try :
            pc = Pinecone(api_key=self.api_key)
            pc.create_index(
                name=index_name,
                dimension=dimentions,
                metric="cosine",
                spec=ServerlessSpec(
                cloud='aws', 
                region='us-east-1'
                ) 
            )
            print(f"Your index {index_name} created successfull")
            return index_name
        except Exception as ex:
            return f"sorry try again {ex}"
        
    # Delete Index Name
    def delete_index_name(self,index_name):
        try:
            pc = Pinecone(api_key=self.api_key)
            indexes = pc.list_indexes().names()
            if index_name not in indexes:
                return f"Index '{index_name}' does not exist."
            pc.delete_index(index_name)
            return f"Index '{index_name}' deleted successfully."
        except Exception as ex:
            return f"Failed to delete index '{index_name}': {ex}"

    # Delete NameSpace 
    def delete_name_spaces(self,index_name,name_space):
        try:
            # Initialize the index
            pc = Pinecone(api_key=self.api_key)
            index = pc.Index(index_name)
            # Delete the namespace
            response = index.delete(namespace=name_space, delete_all=True)
            if response == {}:
                return f"Namespace '{name_space}' deleted successfully from index '{index_name}'."
            else:
                return f"Unexpected response: {response}"
        except Exception :
            return f"An error occurred: Failed to Delete Namespace"
    
    # Create New nameSpace and insert Data in it 
    def insert_data_in_namespace(self,documents,embeddings,index_name,name_space):
        try:
            doc_search=PineconeVectorStore.from_documents(
                documents,
                embeddings,
                index_name=index_name,
                namespace = name_space
                )
            print(f"Your Name space {name_space} is Created successfully")
            return doc_search
        except Exception as ex:
            return f"Failed to created namespace {ex}"
    
    # Insert Data in Index name 
    def insert_data_in_index(self,documents,embeddings,index_name):
        try:
            PineconeVectorStore.from_documents(
                documents,
                embedding=embeddings,
                index_name=index_name
                )
            print(f"Your Data insert in {index_name} successfully")
        except Exception as ex:
            return f"Failed to created namespace {ex}"

    # Retrieve Data from index name 
    def retrieve_from_index_name(self,index_name,embeddings):
        try:
            vectorstore = PineconeVectorStore.from_existing_index(
                embedding=embeddings,index_name=index_name)
            return vectorstore
        except Exception as ex:
            return f"Failed to load VectorStore {ex}"
        
    # Retrieve Data from Namespace
    def retrieve_from_namespace(self,index_name,embeddings,name_space):
        try:
            vectorstore = PineconeVectorStore.from_existing_index(
                embedding=embeddings,index_name=index_name,namespace=name_space)
            return vectorstore
        except Exception as ex:
            return f"Failed to load VectorStore {ex}"


# pine_ = PineconeInsertRetrieval(pinecone_key)


pincoine_ob = PineconeInsertRetrieval(api_key=pinecone_key)