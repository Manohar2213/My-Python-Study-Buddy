"""
Enhanced RAG System using scikit-learn TF-IDF for semantic search
Provides better text matching than simple keyword search
"""

import os
import re
from typing import List, Tuple, Optional

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class EnhancedRAGSystem:
    def __init__(self):
        self.documents = []
        self.document_chunks = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.chunk_size = 500
        self.overlap = 50
        
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\']', ' ', text)
        return text.strip()
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk = ' '.join(chunk_words)
            if len(chunk.strip()) > 50:  # Only keep substantial chunks
                chunks.append(chunk)
                
        return chunks if chunks else [text]
    
    def add_document(self, content: str, filename: str = "Unknown"):
        """Add a document to the knowledge base"""
        try:
            cleaned_content = self.preprocess_text(content)
            chunks = self.chunk_text(cleaned_content)
            
            for chunk in chunks:
                self.documents.append({
                    'content': chunk,
                    'source': filename,
                    'full_content': cleaned_content[:200] + "..." if len(cleaned_content) > 200 else cleaned_content
                })
                self.document_chunks.append(chunk)
            
            return True
        except Exception as e:
            print(f"Error adding document {filename}: {str(e)}")
            return False
    
    def build_index(self):
        """Build TF-IDF index for similarity search"""
        if not self.document_chunks:
            return False
        
        if not SKLEARN_AVAILABLE:
            # Fallback to simple keyword matching
            return True
            
        try:
            # Configure TF-IDF vectorizer with better parameters
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),  # Use both unigrams and bigrams
                min_df=1,
                max_df=0.95,
                lowercase=True,
                analyzer='word'
            )
            
            self.tfidf_matrix = self.vectorizer.fit_transform(self.document_chunks)
            return True
        except Exception as e:
            print(f"Error building index: {str(e)}")
            return False
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float, str]]:
        """Search for relevant documents using TF-IDF similarity or keyword fallback"""
        if not self.document_chunks:
            return []
        
        # If scikit-learn is available, use TF-IDF
        if SKLEARN_AVAILABLE and self.vectorizer and self.tfidf_matrix is not None and self.tfidf_matrix.shape[0] > 0:
            try:
                # Transform query using the same vectorizer
                query_vector = self.vectorizer.transform([self.preprocess_text(query)])
                
                # Calculate cosine similarities
                similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                
                # Get top k most similar documents
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                results = []
                for idx in top_indices:
                    if similarities[idx] > 0.01:  # Only include results with meaningful similarity
                        results.append((
                            self.document_chunks[idx],
                            float(similarities[idx]),
                            self.documents[idx]['source']
                        ))
                
                return results
            except Exception as e:
                print(f"Error in TF-IDF search: {str(e)}")
        
        # Fallback to keyword search
        return self.keyword_search(query, top_k)
    
    def keyword_search(self, query: str, top_k: int = 3) -> List[Tuple[str, float, str]]:
        """Fallback keyword search when TF-IDF is not available"""
        query_words = set(self.preprocess_text(query).lower().split())
        results = []
        
        for i, chunk in enumerate(self.document_chunks):
            chunk_words = set(chunk.lower().split())
            
            # Calculate simple word overlap score
            common_words = query_words.intersection(chunk_words)
            if common_words:
                score = len(common_words) / len(query_words.union(chunk_words))
                results.append((chunk, score, self.documents[i]['source']))
        
        # Sort by score and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def answer_question(self, question: str) -> Tuple[str, List[str]]:
        """Answer a question using retrieved context"""
        # Search for relevant documents
        search_results = self.search(question, top_k=3)
        
        if not search_results:
            return (
                "I couldn't find relevant information in the knowledge base to answer your question. "
                "Try rephrasing your question or asking about basic Python concepts like variables, "
                "functions, loops, or data types.",
                []
            )
        
        # Extract context from search results
        contexts = []
        sources = []
        
        for content, similarity, source in search_results:
            contexts.append(content)
            sources.append(f"{source} (relevance: {similarity:.2f})")
        
        # Generate answer based on retrieved context
        answer = self.generate_answer(question, contexts)
        
        return answer, sources
    
    def generate_answer(self, question: str, contexts: List[str]) -> str:
        """Generate an answer using the retrieved contexts"""
        # Combine contexts
        combined_context = "\n\n".join(contexts)
        
        # Simple template-based answer generation
        question_lower = question.lower()
        
        # Check for specific question types and topics
        if any(word in question_lower for word in ['loop', 'loops', 'iterate', 'iteration', 'for', 'while', 'break', 'continue']):
            answer = f"Here's what you need to know about Python loops:\n\n{combined_context}"
        elif any(word in question_lower for word in ['list', 'lists', 'array', 'append', 'index', 'slice']):
            answer = f"Here's information about Python lists:\n\n{combined_context}"
        elif any(word in question_lower for word in ['dictionary', 'dictionaries', 'dict', 'key', 'value', 'keys', 'values']):
            answer = f"Here's information about Python dictionaries:\n\n{combined_context}"
        elif any(word in question_lower for word in ['string', 'strings', 'text', 'str', 'format', 'split', 'join']):
            answer = f"Here's information about Python strings:\n\n{combined_context}"
        elif any(word in question_lower for word in ['tuple', 'tuples', 'set', 'sets', 'immutable']):
            answer = f"Here's information about Python tuples and sets:\n\n{combined_context}"
        elif any(word in question_lower for word in ['if', 'else', 'elif', 'condition', 'conditional', 'comparison']):
            answer = f"Here's information about Python conditional statements:\n\n{combined_context}"
        elif any(word in question_lower for word in ['function', 'functions', 'def', 'return', 'parameter', 'argument']):
            answer = f"Here's information about Python functions:\n\n{combined_context}"
        elif any(word in question_lower for word in ['module', 'modules', 'import', 'package', 'library']):
            answer = f"Here's information about Python modules and packages:\n\n{combined_context}"
        elif any(word in question_lower for word in ['error', 'exception', 'try', 'except', 'finally', 'raise']):
            answer = f"Here's information about Python error handling:\n\n{combined_context}"
        elif any(word in question_lower for word in ['variable', 'variables', 'data type', 'data types']):
            answer = f"Here's information about Python variables and data types:\n\n{combined_context}"
        elif any(word in question_lower for word in ['what is', 'what are', 'define']):
            answer = f"Based on the Python documentation:\n\n{combined_context}"
        elif any(word in question_lower for word in ['how to', 'how do', 'how can']):
            answer = f"Here's how to do this in Python:\n\n{combined_context}"
        elif any(word in question_lower for word in ['difference', 'compare', 'versus', 'vs']):
            answer = f"Here's the comparison:\n\n{combined_context}"
        elif any(word in question_lower for word in ['example', 'examples']):
            answer = f"Here are some examples:\n\n{combined_context}"
        else:
            answer = f"Regarding your question about Python:\n\n{combined_context}"
        
        # Ensure answer isn't too long
        if len(answer) > 1500:
            answer = answer[:1500] + "\n\n[Answer truncated for brevity]"
        
        return answer


    def build_knowledge_base(self, documents: List[dict]):
        """Build knowledge base from document list (compatibility method)"""
        try:
            for doc in documents:
                self.add_document(doc.get('content', ''), doc.get('source', 'Unknown'))
            
            return self.build_index()
        except Exception as e:
            print(f"Error building knowledge base: {str(e)}")
            return False


def create_enhanced_rag_system() -> EnhancedRAGSystem:
    """Create and return an enhanced RAG system instance"""
    return EnhancedRAGSystem()