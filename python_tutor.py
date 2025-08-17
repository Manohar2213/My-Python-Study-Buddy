import os
from openai import OpenAI
import streamlit as st

class PythonTutor:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        self.model = "gpt-4o"
    
    def get_response(self, question, difficulty="Beginner"):
        """Get a response from the AI tutor for Python programming questions"""
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
    
    def get_step_by_step_solution(self, problem, difficulty="Beginner"):
        """Get a step-by-step solution for a Python programming problem"""
        system_prompt = f"""You are an expert Python programming tutor helping a {difficulty} level student solve a programming problem.

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