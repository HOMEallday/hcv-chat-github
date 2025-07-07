import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import Chroma
# from langchain_community.document_loaders import PDFPlumberLoader # Alternative for more control
# from langchain_community.document_loaders import UnstructuredPDFLoader # Another alternative for scanned PDFs, images

def load_pdfs_from_directory(directory_path: str) -> list:
    """
    Loads all PDF files from a specified directory into LangChain Document objects.

    Args:
        directory_path (str): The path to the directory containing the PDF files.

    Returns:
        list: A list of LangChain Document objects, one for each page in the PDFs.
              Returns an empty list if no PDFs are found or directory is invalid.
    """
    documents = []
    
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at '{directory_path}'")
        return documents

    print(f"Loading PDFs from: {directory_path}")
    
    # Iterate over all files in the given directory
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            print(f"Found PDF: {file_path}")
            try:
                # Use PyPDFLoader for loading PDF documents
                # Each page of the PDF becomes a separate Document in the list
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
                print(f"Successfully loaded {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    if not documents:
        print(f"No PDF documents found or loaded from '{directory_path}'.")
    else:
        print(f"Total documents loaded: {len(documents)} (each representing a PDF page)")
        # Example: print the first 200 characters of the first loaded document
        if documents:
            print(f"First document content snippet: {documents[0].page_content[:200]}...")

    return documents


if __name__ == "__main__":

    my_pdf_directory = "C:\\DevE\\chatHCV3\\full-hcv-1\\data\\raw_documents"
    # Load the documents
    my_documents = load_pdfs_from_directory(my_pdf_directory)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Characters per chunk
        chunk_overlap=200, # Overlap to maintain context between chunks
        separators=["\n\n", "\n", " ", ""] # How to split text
    )
    texts = text_splitter.split_documents(my_documents)
    print(f"Split {len(my_documents)} documents into {len(texts)} chunks.")

    embeddings = VertexAIEmbeddings(model="text-embedding-004")

    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory="./data/chroma_db" #optional 
    )
    print("Documents embedded and stored in vector store.")

