import os
import json
from openai import OpenAI
import streamlit as st

class PythonQuizGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        self.model = "gpt-4o"
    
    def generate_quiz(self, topic="General Python", difficulty="Beginner", num_questions=5):
        """Generate a Python programming quiz"""
        system_prompt = f"""You are an expert Python programming instructor creating a quiz for {difficulty} level students on {topic}.

Create a quiz with {num_questions} questions that test understanding of Python concepts. Mix question types:
- Multiple choice questions about Python concepts
- Code reading/understanding questions
- Short programming challenges

For each question, provide:
- Clear, educational questions appropriate for {difficulty} level
- Multiple choice options (A, B, C, D) when applicable
- Correct answers
- Detailed explanations that help students learn

Return your response in this exact JSON format:
{{
    "title": "Quiz title here",
    "topic": "{topic}",
    "difficulty": "{difficulty}",
    "questions": [
        {{
            "type": "multiple_choice",
            "question": "Question text here",
            "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
            "correct_answer": "A) Option 1",
            "explanation": "Detailed explanation of why this is correct and why other options are wrong"
        }}
    ]
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Create a Python quiz on {topic} for {difficulty} level with {num_questions} questions."}
                ],
                max_tokens=2000,
                temperature=0.8,
                response_format={"type": "json_object"}
            )
            
            quiz_data = json.loads(response.choices[0].message.content)
            return quiz_data
            
        except Exception as e:
            return {
                "title": "Quiz Generation Error",
                "topic": topic,
                "difficulty": difficulty,
                "questions": [{
                    "type": "multiple_choice",
                    "question": f"Sorry, I encountered an error generating the quiz: {str(e)}",
                    "options": ["A) Try again", "B) Try again", "C) Try again", "D) Try again"],
                    "correct_answer": "A) Try again",
                    "explanation": "Please try generating the quiz again."
                }]
            }
    
    def evaluate_quiz(self, quiz, user_answers):
        """Evaluate quiz results and provide feedback"""
        if not quiz or not quiz.get('questions'):
            return {"score": 0, "total": 0, "feedback": []}
        
        results = {
            "score": 0,
            "total": len(quiz["questions"]),
            "feedback": []
        }
        
        for i, question in enumerate(quiz["questions"]):
            user_answer = user_answers.get(f"q_{i}", "")
            correct_answer = question.get("correct_answer", "")
            
            is_correct = user_answer.strip() == correct_answer.strip()
            if is_correct:
                results["score"] += 1
            
            feedback = {
                "question_num": i + 1,
                "correct": is_correct,
                "user_answer": user_answer,
                "correct_answer": correct_answer,
                "explanation": question.get("explanation", "No explanation available.")
            }
            results["feedback"].append(feedback)
        
        return results