# ============================================================
# FILE: triage_ai.py
# ROLE: Data Analyst (DA)
# PURPOSE: Sends patient data to Groq AI and gets triage result.
# NOTE: This file is NOT changed from Day 10.
# ============================================================
 
from groq import Groq
from dotenv import load_dotenv
import os
 
# Load API key from .env file
load_dotenv()
 
# The Groq model we use. This one is free and works on student accounts.
GROQ_MODEL = "openai/gpt-oss-20b"
 
 
def build_prompt(patient: dict) -> str:
    prompt = f"""
You are a medical triage assistant helping a nurse at a hospital outpatient department in India.

A patient has arrived. Based on the information below, provide a brief triage assessment.

Patient Information:
- Name: {patient['name']}
- Age: {patient['age']} years
- Gender: {patient['gender']}
- Reported Symptoms: {patient['symptoms']}
- Blood Pressure: {patient.get('bp', 'Not recorded')}
- Blood Sugar Level: {patient.get('sugar_level', 'Not recorded')}
- Body Temperature: {patient.get('temperature', 'Not recorded')}

Please provide:
1. URGENCY LEVEL: (Emergency / High / Medium / Low) - one word only
2. POSSIBLE CONDITIONS: List 2 to 3 possible medical conditions
3. RECOMMENDED TESTS: List 2 to 3 tests the doctor should order
4. IMMEDIATE ACTION: One sentence on what should happen next

Keep your response clear, concise, and in simple English.
Do not provide a final diagnosis. This is a triage suggestion for the attending nurse.
"""
    return prompt
 
 
def run_triage(patient):
    """
    Sends patient data to Groq AI.
    Returns the AI response as a string.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
 
    prompt = build_prompt(patient)
 
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,    # Low temperature = more predictable, clinical responses
        max_tokens=500,     # Keep responses short
    )
 
    # Extract the AI text from the response object
    return response.choices[0].message.content