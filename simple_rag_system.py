import os
import re
from typing import List, Tuple

class SimpleRAGSystem:
    def __init__(self):
        """Initialize a simple RAG system using keyword matching."""
        self.documents = []
        self.knowledge_base_loaded = False
    
    def build_knowledge_base(self, documents: List[str]):
        """Build the knowledge base from a list of documents."""
        try:
            self.documents = documents
            self.knowledge_base_loaded = True if documents else False
            print(f"Knowledge base built with {len(documents)} documents")
        except Exception as e:
            raise Exception(f"Error building knowledge base: {str(e)}")
    
    def simple_search(self, query: str, documents: List[str], top_k: int = 3) -> List[dict]:
        """Simple keyword-based search for relevant documents."""
        query_words = set(query.lower().split())
        scored_docs = []
        
        for i, doc in enumerate(documents):
            doc_words = set(doc.lower().split())
            # Calculate simple word overlap score
            overlap = len(query_words.intersection(doc_words))
            if overlap > 0:
                scored_docs.append({
                    'content': doc,
                    'score': overlap / len(query_words),
                    'index': i
                })
        
        # Sort by score and return top_k
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        return scored_docs[:top_k]
    
    def generate_simple_answer(self, query: str, relevant_docs: List[dict]) -> str:
        """Generate a simple answer based on relevant documents."""
        if not relevant_docs:
            return "I don't have enough information to answer that question. Please make sure your question is about introductory Python programming concepts."
        
        # Combine the most relevant documents
        context_parts = []
        for doc in relevant_docs[:2]:  # Use top 2 documents
            # Get first few sentences of the document
            sentences = re.split(r'[.!?]+', doc['content'])
            relevant_sentences = [s.strip() for s in sentences[:3] if s.strip()]
            context_parts.extend(relevant_sentences)
        
        # Create a simple response
        context = " ".join(context_parts)
        
        # Simple template-based response
        if "what is" in query.lower() or "what are" in query.lower():
            answer = f"Based on the Python documentation: {context[:500]}..."
        elif "how" in query.lower():
            answer = f"Here's how to work with this in Python: {context[:500]}..."
        else:
            answer = f"Regarding your Python question: {context[:500]}..."
        
        return answer
    
    def answer_question(self, query: str) -> Tuple[str, List[str]]:
        """Main method to answer a question using simple RAG."""
        try:
            # Validate inputs
            if not query.strip():
                return "Please ask a question about Python programming.", []
            
            if not self.knowledge_base_loaded:
                return "Knowledge base not initialized. Please load documents first.", []
            
            # Find relevant documents using simple search
            relevant_docs = self.simple_search(query, self.documents)
            
            if not relevant_docs:
                return "I couldn't find relevant information for your question. Try asking about Python basics, functions, or data types.", []
            
            # Generate simple answer
            answer = self.generate_simple_answer(query, relevant_docs)
            
            # Extract sources
            sources = [doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'] 
                      for doc in relevant_docs[:3]]
            
            return answer, sources
            
        except Exception as e:
            return f"Error processing question: {str(e)}", []