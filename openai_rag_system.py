import os
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

class OpenAIRAGSystem:
    def __init__(self):
        # Initialize OpenAI client
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = "gpt-4o"
        
        # Initialize sentence transformer for embeddings
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Storage for documents and embeddings
        self.documents = []
        self.document_sources = []
        self.embeddings = None
        self.index = None
        
    def add_documents(self, documents: List[str], sources: List[str] = None):
        """Add documents to the knowledge base"""
        if sources is None:
            sources = [f"Document {i}" for i in range(len(documents))]
        
        # Add to existing documents
        self.documents.extend(documents)
        self.document_sources.extend(sources)
        
        # Generate embeddings for all documents
        all_embeddings = self.embedding_model.encode(self.documents)
        self.embeddings = np.array(all_embeddings).astype('float32')
        
        # Create or update FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        print(f"Added {len(documents)} documents to knowledge base. Total: {len(self.documents)}")
    
    def retrieve_relevant_contexts(self, question: str, top_k: int = 3) -> Tuple[List[str], List[str]]:
        """Retrieve most relevant document chunks for the question"""
        if self.index is None or len(self.documents) == 0:
            return [], []
        
        # Encode the question
        question_embedding = self.embedding_model.encode([question])
        question_embedding = np.array(question_embedding).astype('float32')
        faiss.normalize_L2(question_embedding)
        
        # Search for similar documents
        scores, indices = self.index.search(question_embedding, min(top_k, len(self.documents)))
        
        # Get the relevant contexts and sources
        contexts = []
        sources = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and scores[0][i] > 0.1:  # Minimum similarity threshold
                contexts.append(self.documents[idx])
                sources.append(self.document_sources[idx])
        
        return contexts, sources
    
    def generate_answer_with_openai(self, question: str, contexts: List[str]) -> str:
        """Generate an answer using OpenAI GPT-4o with retrieved contexts"""
        if not contexts:
            return "I don't have enough information in my knowledge base to answer that question. Please try asking about Python basics, functions, loops, data structures, conditionals, modules, or error handling."
        
        # Combine contexts
        combined_context = "\n\n".join(contexts)
        
        # Create a comprehensive prompt for the AI
        system_prompt = """You are a helpful Python programming tutor. Your job is to answer student questions about Python programming using the provided knowledge base content.

Instructions:
- Use the context information provided to give accurate, educational answers
- Explain concepts clearly for students learning Python
- Include practical examples when helpful
- If the context doesn't fully answer the question, acknowledge what you can answer and suggest related topics
- Keep answers focused and educational
- Use simple, clear language appropriate for Python learners
"""

        user_prompt = f"""Context from Python knowledge base:
{combined_context}

Student question: {question}

Please provide a helpful, educational answer based on the context provided. Include examples when they help explain the concept."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"I encountered an error while generating the response: {str(e)}. Please try asking your question again."
    
    def answer_question(self, question: str) -> Tuple[str, List[str]]:
        """Main method to answer questions using RAG with OpenAI"""
        # Retrieve relevant contexts
        contexts, sources = self.retrieve_relevant_contexts(question, top_k=3)
        
        # Generate answer using OpenAI
        answer = self.generate_answer_with_openai(question, contexts)
        
        return answer, sources
    
    def get_knowledge_base_stats(self) -> dict:
        """Get statistics about the current knowledge base"""
        return {
            "total_documents": len(self.documents),
            "embedding_dimension": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "sources": list(set(self.document_sources))
        }