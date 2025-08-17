import os
from openai import OpenAI
import streamlit as st
from typing import List, Optional
import numpy as np

# Try to import RAG dependencies
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

RAG_DEPENDENCIES_AVAILABLE = SENTENCE_TRANSFORMERS_AVAILABLE and FAISS_AVAILABLE

class HybridPythonTutor:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        self.model = "gpt-4o"
        
        # RAG components
        self.embedding_model = None
        self.knowledge_base = []
        self.embeddings = None
        self.index = None
        self.rag_enabled = False
    
    def initialize_rag(self):
        """Initialize RAG components"""
        if not RAG_DEPENDENCIES_AVAILABLE:
            st.error("RAG dependencies not available. Please install sentence-transformers and faiss-cpu.")
            return False
        
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.rag_enabled = True
            return True
        except Exception as e:
            st.error(f"Failed to initialize RAG: {str(e)}")
            return False
    
    def add_documents(self, documents: List[str], sources: List[str]):
        """Add documents to the knowledge base"""
        if not self.rag_enabled:
            return False
        
        try:
            # Store documents with sources
            for doc, source in zip(documents, sources):
                self.knowledge_base.append({
                    'content': doc,
                    'source': source
                })
            
            # Generate embeddings
            all_texts = [doc['content'] for doc in self.knowledge_base]
            self.embeddings = self.embedding_model.encode(all_texts)
            
            # Build FAISS index
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(self.embeddings.astype('float32'))
            
            return True
        except Exception as e:
            st.error(f"Error adding documents: {str(e)}")
            return False
    
    def retrieve_relevant_docs(self, query: str, top_k: int = 3) -> List[dict]:
        """Retrieve relevant documents for a query"""
        if not self.rag_enabled or self.index is None:
            return []
        
        try:
            # Encode query
            query_embedding = self.embedding_model.encode([query])
            
            # Search for similar documents
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            # Return relevant documents
            relevant_docs = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.knowledge_base):
                    doc = self.knowledge_base[idx].copy()
                    doc['similarity_score'] = float(score)
                    relevant_docs.append(doc)
            
            return relevant_docs
        except Exception as e:
            st.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def get_response(self, question: str, difficulty: str = "Beginner", use_rag: bool = False) -> str:
        """Get response with optional RAG enhancement"""
        
        if use_rag and self.rag_enabled:
            return self._get_rag_response(question, difficulty)
        else:
            return self._get_simple_response(question, difficulty)
    
    def _get_simple_response(self, question: str, difficulty: str) -> str:
        """Get simple AI response without RAG"""
        system_prompt = f"""You are an expert Python programming tutor. Your student is at {difficulty} level.

Your role is to:
1. Provide clear, educational answers about Python programming
2. Use practical code examples to illustrate concepts
3. Explain concepts step-by-step for {difficulty} level students
4. Encourage best practices and good coding habits
5. Be patient and supportive in your explanations

Always include:
- Clear explanations appropriate for {difficulty} level
- Working code examples when relevant
- Common mistakes to avoid
- Tips for better programming practices

Keep responses educational, encouraging, and focused on helping the student learn Python effectively."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"I encountered an error: {str(e)}. Please try asking your question again."
    
    def _get_rag_response(self, question: str, difficulty: str) -> str:
        """Get RAG-enhanced response"""
        # Retrieve relevant documents
        relevant_docs = self.retrieve_relevant_docs(question, top_k=3)
        
        # Build context from retrieved documents
        context = ""
        sources = []
        if relevant_docs:
            context = "Here is relevant information from the knowledge base:\n\n"
            for i, doc in enumerate(relevant_docs):
                context += f"Context {i+1} (from {doc['source']}):\n{doc['content']}\n\n"
                sources.append(doc['source'])
        
        system_prompt = f"""You are an expert Python programming tutor with access to a comprehensive knowledge base. Your student is at {difficulty} level.

Your role is to:
1. Use the provided context to give accurate, detailed answers about Python programming
2. Combine knowledge base information with your expertise
3. Provide practical code examples and explanations
4. Adapt explanations for {difficulty} level students
5. Be educational and supportive

When using the knowledge base:
- Reference the context provided below when relevant
- Expand on the information with additional examples and explanations
- If the context doesn't fully answer the question, use your knowledge to provide a complete response

{context}

Guidelines:
- Give clear explanations appropriate for {difficulty} level
- Include working code examples when helpful
- Mention common mistakes to avoid
- Provide tips for better programming practices
- If you used information from the knowledge base, mention that you found relevant information in the study materials"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                max_tokens=1200,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Add source information if RAG was used
            if sources:
                unique_sources = list(set(sources))
                answer += f"\n\n*Based on information from: {', '.join(unique_sources)}*"
            
            return answer
            
        except Exception as e:
            return f"I encountered an error: {str(e)}. Please try asking your question again."
    
    def get_step_by_step_solution(self, problem: str, difficulty: str = "Beginner", use_rag: bool = False) -> str:
        """Get step-by-step solution with optional RAG enhancement"""
        
        if use_rag and self.rag_enabled:
            # Retrieve relevant documents for the problem
            relevant_docs = self.retrieve_relevant_docs(problem, top_k=2)
            
            context = ""
            if relevant_docs:
                context = "Relevant information from knowledge base:\n\n"
                for doc in relevant_docs:
                    context += f"From {doc['source']}: {doc['content']}\n\n"
        else:
            context = ""
        
        system_prompt = f"""You are an expert Python programming tutor helping a {difficulty} level student solve a programming problem step by step.

{context}

Break down the solution into clear, logical steps:
1. Understand the problem and what needs to be accomplished
2. Plan the approach and identify key Python concepts needed
3. Write the code step by step with explanations
4. Show the complete working solution
5. Explain how the solution works
6. Provide tips for similar problems

Use proper Python syntax and include comments in your code examples."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Help me solve this Python problem step by step: {problem}"}
                ],
                max_tokens=1200,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"I encountered an error: {str(e)}. Please try asking again."