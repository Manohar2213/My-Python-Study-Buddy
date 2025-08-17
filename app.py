import streamlit as st
import os
import tempfile
from python_tutor import PythonTutor
from python_quiz_generator import PythonQuizGenerator
from hybrid_tutor import HybridPythonTutor
from document_processor import DocumentProcessor

# Page config
st.set_page_config(
    page_title="Python Programming Tutor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def initialize_session_state():
    if "questions_asked" not in st.session_state:
        st.session_state.questions_asked = 0
    if "problems_solved" not in st.session_state:
        st.session_state.problems_solved = 0
    if "quizzes_completed" not in st.session_state:
        st.session_state.quizzes_completed = 0
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_quiz" not in st.session_state:
        st.session_state.current_quiz = None
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}
    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False
    if "performance_history" not in st.session_state:
        st.session_state.performance_history = []
    if "rag_mode" not in st.session_state:
        st.session_state.rag_mode = False
    if "hybrid_tutor" not in st.session_state:
        st.session_state.hybrid_tutor = None
    if "knowledge_base_loaded" not in st.session_state:
        st.session_state.knowledge_base_loaded = False

def validate_openai_key():
    """Validate that OpenAI API key is available"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found. Please add your OPENAI_API_KEY to environment variables.")
        st.stop()
    return True

def render_progress_dashboard():
    """Render the progress tracking dashboard"""
    st.subheader("Your Learning Progress")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Questions Asked", st.session_state.questions_asked)
    
    with col2:
        st.metric("Problems Solved", st.session_state.problems_solved)
    
    with col3:
        st.metric("Quizzes Completed", st.session_state.quizzes_completed)
    
    if st.session_state.performance_history:
        avg_score = sum(st.session_state.performance_history) / len(st.session_state.performance_history)
        st.metric("Average Quiz Score", f"{avg_score:.1f}%")

def main():
    initialize_session_state()
    validate_openai_key()
    
    # Initialize components
    python_tutor = PythonTutor()
    quiz_generator = PythonQuizGenerator()
    
    # Initialize hybrid tutor for RAG capability
    if st.session_state.hybrid_tutor is None:
        st.session_state.hybrid_tutor = HybridPythonTutor()
    
    # Custom CSS for dark theme
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #2a5298;
        background-color: #f8f9fa;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #1976d2;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left-color: #7b1fa2;
    }
    .stButton > button {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header"><h1>Python Programming Tutor</h1><p>Learn Python with AI-powered guidance</p></div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Learning Tools")
        
        # Difficulty selector
        difficulty = st.selectbox(
            "Select Your Level",
            ["Beginner", "Intermediate", "Advanced"],
            key="difficulty"
        )
        
        st.markdown("---")
        
        # Learning mode selector
        mode = st.radio(
            "Choose Learning Mode",
            ["Ask Questions", "Solve Problems", "Take Quiz", "View Progress"]
        )
        
        st.markdown("---")
        
        # RAG Enhancement Section
        st.markdown("---")
        st.subheader("Enhanced Learning")
        
        # RAG Mode Toggle
        rag_enabled = st.checkbox(
            "Enable Knowledge Base (RAG)",
            value=st.session_state.rag_mode,
            help="Use additional study materials for enhanced answers"
        )
        
        if rag_enabled != st.session_state.rag_mode:
            st.session_state.rag_mode = rag_enabled
        
        if rag_enabled:
            # Check if RAG dependencies are available
            from hybrid_tutor import RAG_DEPENDENCIES_AVAILABLE
            if not RAG_DEPENDENCIES_AVAILABLE:
                st.warning("RAG functionality requires additional packages: sentence-transformers and faiss-cpu")
                st.info("The app will work in simple mode. To enable RAG, install the required packages.")
                st.session_state.rag_mode = False
            else:
                # Initialize RAG if not already done
                if not st.session_state.hybrid_tutor.rag_enabled:
                    with st.spinner("Initializing knowledge base..."):
                        if st.session_state.hybrid_tutor.initialize_rag():
                            st.success("Knowledge base ready!")
                        else:
                            st.error("Failed to initialize knowledge base")
                            st.session_state.rag_mode = False
            
            # Knowledge base status
            if st.session_state.knowledge_base_loaded:
                st.success("Knowledge base loaded")
            else:
                st.info("Upload documents to enhance responses")
            
            # File upload for knowledge base
            uploaded_files = st.file_uploader(
                "Upload Python study materials",
                type=['txt', 'pdf', 'md'],
                accept_multiple_files=True,
                key="knowledge_upload"
            )
            
            if uploaded_files and not st.session_state.knowledge_base_loaded:
                if st.button("Load Documents"):
                    with st.spinner("Processing documents..."):
                        doc_processor = DocumentProcessor()
                        all_documents = []
                        all_sources = []
                        
                        for uploaded_file in uploaded_files:
                            # Save uploaded file temporarily
                            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_file_path = tmp_file.name
                            
                            try:
                                chunks = doc_processor.process_file(tmp_file_path)
                                all_documents.extend(chunks)
                                all_sources.extend([uploaded_file.name] * len(chunks))
                            finally:
                                os.unlink(tmp_file_path)
                        
                        if all_documents:
                            if st.session_state.hybrid_tutor.add_documents(all_documents, all_sources):
                                st.session_state.knowledge_base_loaded = True
                                st.success(f"Loaded {len(all_documents)} document chunks from {len(uploaded_files)} files")
                            else:
                                st.error("Failed to load documents")
                        else:
                            st.warning("No valid documents found")
            
            if st.session_state.knowledge_base_loaded and st.button("Clear Knowledge Base"):
                st.session_state.hybrid_tutor = HybridPythonTutor()
                st.session_state.knowledge_base_loaded = False
                st.info("Knowledge base cleared")
        
        st.markdown("---")
        
        # Python topics
        st.subheader("Python Topics")
        topics = [
            "Variables & Data Types",
            "Control Flow (if/else)",
            "Loops (for/while)",
            "Functions",
            "Lists & Arrays",
            "Dictionaries",
            "Strings",
            "File Handling",
            "Error Handling",
            "Object-Oriented Programming",
            "Modules & Packages"
        ]
        
        for topic in topics:
            st.markdown(f"• {topic}")
    
    # Main content area
    if mode == "Ask Questions":
        st.subheader("Ask Python Programming Questions")
        st.write("Ask me anything about Python programming and I'll provide detailed explanations!")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask your Python question..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get AI response (RAG-enhanced if enabled)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    if st.session_state.rag_mode and st.session_state.hybrid_tutor.rag_enabled:
                        response = st.session_state.hybrid_tutor.get_response(prompt, difficulty, use_rag=True)
                    else:
                        response = python_tutor.get_response(prompt, difficulty)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.questions_asked += 1
    
    elif mode == "Solve Problems":
        st.subheader("Step-by-Step Problem Solving")
        st.write("Describe a Python problem and I'll walk you through the solution step by step.")
        
        problem_text = st.text_area(
            "Describe your Python problem:",
            placeholder="Example: Create a function that finds the largest number in a list",
            height=100
        )
        
        if st.button("Get Step-by-Step Solution"):
            if problem_text:
                with st.spinner("Generating solution..."):
                    if st.session_state.rag_mode and st.session_state.hybrid_tutor.rag_enabled:
                        solution = st.session_state.hybrid_tutor.get_step_by_step_solution(problem_text, difficulty, use_rag=True)
                    else:
                        solution = python_tutor.get_step_by_step_solution(problem_text, difficulty)
                    st.markdown("### Solution:")
                    st.markdown(solution)
                    st.session_state.problems_solved += 1
            else:
                st.warning("Please describe a problem first!")
    
    elif mode == "Take Quiz":
        st.subheader("Python Programming Quiz")
        
        if st.session_state.current_quiz is None:
            st.write("Test your Python knowledge with an adaptive quiz!")
            
            col1, col2 = st.columns(2)
            with col1:
                topic = st.selectbox(
                    "Quiz Topic",
                    ["General Python", "Variables & Data Types", "Control Flow", "Functions", 
                     "Lists & Dictionaries", "String Operations", "Error Handling"]
                )
            
            with col2:
                num_questions = st.slider("Number of Questions", 3, 10, 5)
            
            if st.button("Generate Quiz"):
                with st.spinner("Creating your quiz..."):
                    quiz = quiz_generator.generate_quiz(topic, difficulty, num_questions)
                    st.session_state.current_quiz = quiz
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_submitted = False
                    st.rerun()
        
        else:
            quiz = st.session_state.current_quiz
            st.write(f"**{quiz['title']}**")
            st.write(f"Topic: {quiz['topic']} | Difficulty: {quiz['difficulty']}")
            
            if not st.session_state.quiz_submitted:
                # Display quiz questions
                for i, question in enumerate(quiz["questions"]):
                    st.markdown(f"### Question {i+1}")
                    st.markdown(question["question"])
                    
                    if question["type"] == "multiple_choice":
                        answer = st.radio(
                            f"Select your answer for Question {i+1}:",
                            question["options"],
                            key=f"q_{i}"
                        )
                        st.session_state.quiz_answers[f"q_{i}"] = answer
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Submit Quiz"):
                        results = quiz_generator.evaluate_quiz(quiz, st.session_state.quiz_answers)
                        st.session_state.quiz_results = results
                        st.session_state.quiz_submitted = True
                        st.session_state.quizzes_completed += 1
                        
                        # Track performance
                        score_percentage = (results["score"] / results["total"]) * 100
                        st.session_state.performance_history.append(score_percentage)
                        st.rerun()
                
                with col2:
                    if st.button("New Quiz"):
                        st.session_state.current_quiz = None
                        st.session_state.quiz_answers = {}
                        st.session_state.quiz_submitted = False
                        st.rerun()
            
            else:
                # Display results
                results = st.session_state.quiz_results
                score_percentage = (results["score"] / results["total"]) * 100
                
                st.success(f"Quiz Complete! Your Score: {results['score']}/{results['total']} ({score_percentage:.1f}%)")
                
                # Show detailed feedback
                st.subheader("Detailed Feedback")
                for feedback in results["feedback"]:
                    with st.expander(f"Question {feedback['question_num']} - {'Correct' if feedback['correct'] else 'Incorrect'}"):
                        st.write(f"**Your answer:** {feedback['user_answer']}")
                        st.write(f"**Correct answer:** {feedback['correct_answer']}")
                        st.write(f"**Explanation:** {feedback['explanation']}")
                
                if st.button("Take Another Quiz"):
                    st.session_state.current_quiz = None
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_submitted = False
                    st.rerun()
    
    elif mode == "View Progress":
        st.subheader("Learning Analytics")
        render_progress_dashboard()
        
        if st.session_state.performance_history:
            st.subheader("Quiz Performance Over Time")
            st.line_chart(st.session_state.performance_history)
            
            st.subheader("Performance Analysis")
            avg_score = sum(st.session_state.performance_history) / len(st.session_state.performance_history)
            
            if avg_score >= 80:
                st.success("Excellent performance! You're mastering Python concepts well.")
            elif avg_score >= 60:
                st.info("Good progress! Keep practicing to improve your understanding.")
            else:
                st.warning("Consider reviewing the basics and taking more practice quizzes.")
        else:
            st.info("Take some quizzes to see your performance analytics!")

if __name__ == "__main__":
    main()