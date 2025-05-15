#!/usr/bin/env python3
"""
RAG (Retrieval Augmented Generation) System
-------------------------------------------
This module implements a RAG system that processes PDF documents,
uses ChromaDB as a vector database, sentence-transformers for embeddings,
and Google's Gemini as the main LLM. The system follows a conversational pattern.
"""

import os
import logging
from typing import List, Dict, Any, Optional

# Document processing
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embeddings
from sentence_transformers import SentenceTransformer

# Vector database
import chromadb
from chromadb.utils import embedding_functions

# For Gemini LLM integration
from gemini_wrapper import GoogleGeminiWrapper

from gtts import gTTS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGSystem:
    """
    A Retrieval Augmented Generation system that processes PDF documents,
    stores their embeddings in a vector database, and generates responses
    using the Google Gemini model.
    """
    
    def __init__(
        self, 
        pdf_dir: str, 
        gemini_api_key: str,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        db_directory: str = "./chroma_db"
    ):
        """
        Initialize the RAG system.
        
        Args:
            pdf_dir: Directory containing PDF documents
            gemini_api_key: API key for Google Gemini
            embedding_model_name: Name of the sentence-transformers model
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between consecutive chunks
            db_directory: Directory to store the ChromaDB database
        """
        self.pdf_dir = pdf_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.db_directory = db_directory
        
        # Initialize the embedding model
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        
        # Initialize ChromaDB
        logger.info(f"Initializing ChromaDB at {db_directory}")
        self.client = chromadb.PersistentClient(path=db_directory)
        
        # Create a custom embedding function that uses sentence-transformers
        self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )
        
        # Create or get the collection
        self.collection = self.client.get_or_create_collection(
            name="pdf_documents",
            embedding_function=self.sentence_transformer_ef
        )
        
        # Initialize the Gemini LLM
        logger.info("Initializing Google Gemini")
        self.llm = GoogleGeminiWrapper(api_key=gemini_api_key)
        
        # Load conversation history
        self.conversation_history = []
        
    def process_documents(self) -> None:
        """
        Process all PDF documents in the specified directory,
        split them into chunks, generate embeddings, and store in ChromaDB.
        """
        logger.info(f"Processing documents from: {self.pdf_dir}")
        
        # Check if documents are already processed
        if self.collection.count() > 0:
            logger.info(f"Found {self.collection.count()} existing document chunks in the database")
            return
        
        # Process each PDF file in the directory
        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.endswith('.pdf')]
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.pdf_dir}")
            return
            
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        doc_chunks = []
        metadatas = []
        ids = []
        chunk_idx = 0
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.pdf_dir, pdf_file)
            logger.info(f"Processing: {pdf_path}")
            
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Split {pdf_file} into {len(chunks)} chunks")
            
            # Prepare data for ChromaDB
            for chunk in chunks:
                doc_chunks.append(chunk.page_content)
                metadatas.append({
                    "source": pdf_file,
                    "page": chunk.metadata.get("page", 0),
                })
                ids.append(f"chunk_{chunk_idx}")
                chunk_idx += 1
        
        # Add documents to ChromaDB
        if doc_chunks:
            logger.info(f"Adding {len(doc_chunks)} chunks to ChromaDB")
            self.collection.add(
                documents=doc_chunks,
                metadatas=metadatas,
                ids=ids
            )
            logger.info("Documents successfully processed and stored")
        else:
            logger.warning("No document chunks were generated")
            
    def retrieve_relevant_chunks(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve the k most relevant document chunks for a given query.
        
        Args:
            query: The query text
            k: Number of relevant chunks to retrieve
            
        Returns:
            List of relevant document chunks with their metadata
        """
        logger.info(f"Retrieving {k} relevant chunks for query: {query}")
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        relevant_chunks = []
        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                relevant_chunks.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else {},
                    "id": results["ids"][0][i] if results["ids"] and results["ids"][0] else f"unknown_{i}"
                })
        
        return relevant_chunks
    
    def generate_response(self, query: str, k: int = 3) -> str:
        """
        Generate a response for a user query using RAG.
        
        Args:
            query: User query
            k: Number of relevant chunks to retrieve
            
        Returns:
            Generated response from the LLM
        """
        # Retrieve relevant document chunks
        relevant_chunks = self.retrieve_relevant_chunks(query, k=k)
        
        if not relevant_chunks:
            logger.warning("No relevant chunks found for the query")
            return "I couldn't find relevant information to answer your question."
        
        # Format context from retrieved chunks
        context = "\n\n".join([f"Document {i+1} (from {chunk['metadata'].get('source', 'unknown')}, page {chunk['metadata'].get('page', 'unknown')}):\n{chunk['content']}" 
                              for i, chunk in enumerate(relevant_chunks)])
        
        # Create prompt for the LLM
        prompt = f"""
        You are a helpful assistant that answers questions based on the provided context.
        
        CONTEXT:
        {context}
        
        QUESTION:
        {query}
        
        Please provide a comprehensive and accurate answer based only on the information in the provided context.
        If the context doesn't contain enough information to answer the question, please say so.
        """
        
        # Generate response using Gemini
        response = self.llm.ask(prompt, max_tokens=500, temperature=0.3)
        return response
    
    def chat(self, user_input: str = None) -> Optional[str]:
        """
        Conduct a conversation with the user using the RAG system.
        
        Args:
            user_input: User's input. If None, starts a new conversation.
            
        Returns:
            System's response or None to exit
        """
        if user_input is None:
            # Initialize conversation
            print("RAG System Initialized. Type 'exit' or 'quit' to end the conversation.")
            user_input = input("You: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Ending conversation. Goodbye!")
            return None
        
        # Generate response using RAG
        response = self.generate_response(user_input)
        
        # Update conversation history
        self.conversation_history.append({"user": user_input, "system": response})
        
        return response
    
    def interactive_session(self) -> None:
        """
        Start an interactive chat session with the RAG system.
        """
        print("Welcome to the RAG System!")
        print("Type 'exit' or 'quit' to end the conversation.")
        
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ['exit', 'quit']:
                print("Ending conversation. Goodbye!")
                break
            
            response = self.generate_response(user_input)
            print(f"\nRAG System: {response}")

# Function to convert text to speech
def text_to_speech(response):
    tts = gTTS(response)
    audio_path = "response_audio.mp3"
    tts.save(audio_path)
    return audio_path

def main():
    """
    Main function to demonstrate the RAG system.
    """
    # Attempt to get the Gemini API key from environment variable
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not gemini_api_key:
        # If environment variable is not set or is empty, fallback to the hardcoded key
        hardcoded_api_key = "GEMINI_API_KEY" # Your hardcoded key
        # Check if the environment variable was truly not set (vs. set to an empty string)
        # to decide if we should print the INFO message.
        if os.getenv("GEMINI_API_KEY") is None: # More specific check for unset env variable
            print("INFO: GEMINI_API_KEY environment variable not found. Using hardcoded API key from rag.py.")
        gemini_api_key = hardcoded_api_key
    
    # Final check: if the key is still not set (e.g. if hardcoded key was also empty or None)
    if not gemini_api_key:
        print("Error: Gemini API key is not set.")
        print("Please set the GEMINI_API_KEY environment variable, or ensure it's correctly hardcoded in rag.py.")
        print("To set as environment variable:")
        print("  export GEMINI_API_KEY='your_api_key'  # For Linux/macOS")
        print("  set GEMINI_API_KEY=your_api_key      # For Windows CMD")
        print("  $env:GEMINI_API_KEY='your_api_key'   # For Windows PowerShell")
        return
    
    # Set paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_dir = os.path.join(current_dir, "material")
    db_dir = os.path.join(current_dir, "chroma_db")
    
    # Initialize the RAG system
    rag = RAGSystem(
        pdf_dir=pdf_dir,
        gemini_api_key=gemini_api_key,
        db_directory=db_dir
    )
    
    # Process documents
    rag.process_documents()
    
    # Start interactive session
    rag.interactive_session()


if __name__ == "__main__":
    main()
