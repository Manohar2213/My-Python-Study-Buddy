import os
import re
from typing import List
import streamlit as st

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.warning("PyPDF2 not available. PDF processing will be disabled.")

class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        if not text:
            return []
        
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk = ' '.join(chunk_words)
            chunks.append(chunk)
            
            # Move start position considering overlap
            start = end - self.overlap
            if start >= len(words):
                break
        
        return chunks
    
    def process_text_file(self, file_path: str) -> List[str]:
        """Process a text file and return chunks"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Clean the text
            cleaned_content = self.clean_text(content)
            
            # Split into chunks
            chunks = self.chunk_text(cleaned_content)
            
            return chunks
            
        except Exception as e:
            st.error(f"Error processing text file {file_path}: {str(e)}")
            return []
    
    def process_pdf(self, file_path: str) -> List[str]:
        """Process a PDF file and return chunks"""
        if not PDF_AVAILABLE:
            st.error("PDF processing is not available. Please install PyPDF2.")
            return []
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract text from all pages
                full_text = ""
                for page in pdf_reader.pages:
                    full_text += page.extract_text() + "\n"
            
            # Clean the text
            cleaned_content = self.clean_text(full_text)
            
            # Split into chunks
            chunks = self.chunk_text(cleaned_content)
            
            return chunks
            
        except Exception as e:
            st.error(f"Error processing PDF file {file_path}: {str(e)}")
            return []
    
    def process_file(self, file_path: str) -> List[str]:
        """Process any supported file type"""
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            return []
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return self.process_pdf(file_path)
        elif file_extension in ['.txt', '.md']:
            return self.process_text_file(file_path)
        else:
            st.warning(f"Unsupported file type: {file_extension}")
            return []