import os
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sklearn.metrics.pairwise import cosine_similarity
import torch

class RAGSystem:
    def __init__(self):
        """Initialize the RAG system with embedding and generation models."""
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.generation_model_name = "google/flan-t5-small"
        
        # Initialize models
        self.embedding_model = None
        self.generation_model = None
        self.tokenizer = None
        
        # Knowledge base components
        self.documents = []
        self.embeddings = None
        
        self._load_models()
    
    def _load_models(self):
        """Load the embedding and generation models."""
        try:
            # Load embedding model
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            # Load generation model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.generation_model_name)
            self.generation_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.generation_model_name,
                torch_dtype=torch.float32
            )
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text2text-generation",
                model=self.generation_model,
                tokenizer=self.tokenizer,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
        except Exception as e:
            raise Exception(f"Error loading models: {str(e)}")
    
    def build_knowledge_base(self, documents):
        """Build the knowledge base from a list of documents."""
        try:
            self.documents = documents
            
            if not documents:
                raise ValueError("No documents provided for knowledge base")
            
            # Generate embeddings for all documents
            document_texts = [doc for doc in documents]
            self.embeddings = self.embedding_model.encode(document_texts)
            
        except Exception as e:
            raise Exception(f"Error building knowledge base: {str(e)}")
    
    def retrieve_relevant_documents(self, query, top_k=3):
        """Retrieve the most relevant documents for a given query."""
        try:
            if self.embeddings is None:
                raise ValueError("Knowledge base not built. Please build knowledge base first.")
            
            # Generate embedding for the query
            query_embedding = self.embedding_model.encode([query])
            
            # Calculate cosine similarity between query and all documents
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top_k most similar documents
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Retrieve relevant documents
            relevant_docs = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Only include documents with reasonable similarity
                    relevant_docs.append({
                        'content': self.documents[idx],
                        'score': float(similarities[idx]),
                        'index': int(idx)
                    })
            
            return relevant_docs
            
        except Exception as e:
            raise Exception(f"Error retrieving documents: {str(e)}")
    
    def generate_answer(self, query, relevant_docs):
        """Generate an answer based on the query and relevant documents."""
        try:
            if not relevant_docs:
                return "I don't have enough information to answer that question. Please make sure your question is about introductory Python programming concepts.", []
            
            # Prepare context from relevant documents
            context_parts = []
            for doc in relevant_docs:
                context_parts.append(doc['content'])
            
            context = "\n\n".join(context_parts)
            
            # Create prompt for the language model
            prompt = f"""Answer the following question about Python programming based on the provided context. 
            Give a clear, concise, and educational explanation suitable for beginners.

Context:
{context}

Question: {query}

Answer:"""
            
            # Generate answer using the language model
            response = self.generator(prompt, max_length=512, truncation=True)
            answer = response[0]['generated_text']
            
            # Extract source information
            sources = [doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'] 
                      for doc in relevant_docs]
            
            return answer, sources
            
        except Exception as e:
            return f"Error generating answer: {str(e)}", []
    
    def answer_question(self, query):
        """Main method to answer a question using RAG."""
        try:
            # Validate inputs
            if not query.strip():
                return "Please ask a question about Python programming.", []
            
            if self.embeddings is None:
                return "Knowledge base not initialized. Please load documents first.", []
            
            # Retrieve relevant documents
            relevant_docs = self.retrieve_relevant_documents(query, top_k=3)
            
            # Generate answer
            answer, sources = self.generate_answer(query, relevant_docs)
            
            return answer, sources
            
        except Exception as e:
            return f"Error processing question: {str(e)}", []
