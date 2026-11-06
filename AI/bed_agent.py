# ============================================================
# FILE: bed_agent.py
# ROLE: Data Analyst (DA)
# PURPOSE: AI Agent that manages hospital bed assignments and discharges.
#
# HOW TO RUN: python bed_agent.py
# ============================================================

import json
import requests      # For calling FastAPI tool endpoints
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# ============================================================
# STEP 1: Define the tools for Groq AI
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_bed",
            "description": "Check if there is an empty bed in a hospital ward.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ward": {
                        "type": "string",
                        "description": "Ward name. Options: General, Emergency, ICU, Pediatric, Maternity"
                    }
                },
                "required": ["ward"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_on_duty_nurse",
            "description": "Get the name of the nurse currently on duty in a ward.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ward": {
                        "type": "string",
                        "description": "Ward name"
                    }
                },
                "required": ["ward"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "assign_bed",
            "description": "Assign a patient to a hospital bed with a nurse.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "integer",
                        "description": "The patient ID number"
                    },
                    "bed_id": {
                        "type": "integer",
                        "description": "The bed ID number"
                    },
                    "nurse_name": {
                        "type": "string",
                        "description": "The name of the nurse who will handle this patient"
                    }
                },
                "required": ["patient_id", "bed_id", "nurse_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "discharge_patient",
            "description": "Discharge a patient from a bed, set bed as unassigned/empty, and log the discharge.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "integer",
                        "description": "The patient ID number to discharge"
                    },
                    "bed_id": {
                        "type": "integer",
                        "description": "The bed ID number being freed"
                    }
                },
                "required": ["patient_id", "bed_id"]
            }
        }
    }
]


# ============================================================
# STEP 2: Function to call a tool on FastAPI
# ============================================================

FASTAPI_URL = "http://localhost:8000"   # FastAPI runs here


def call_tool(tool_name: str, tool_args: dict) -> str:
    """
    Calls one of our FastAPI MCP tool endpoints.
    Returns the result as a JSON string.
    """
    url = f"{FASTAPI_URL}/tools/{tool_name}"

    try:
        response = requests.post(url, json=tool_args)
        result = response.json()
        print(f"  [TOOL CALLED] {tool_name}({tool_args})")
        print(f"  [TOOL RESULT] {result}")
        return json.dumps(result)

    except Exception as e:
        error_msg = f"Tool call failed: {str(e)}"
        print(f"  [TOOL ERROR] {error_msg}")
        return json.dumps({"error": error_msg})


# ============================================================
# STEP 3: The Agent Loops
# ============================================================

def execute_agent_loop(task_message: str):
    """
    Generic loop to send instructions to Groq AI and execute tools.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    messages = [
        {"role": "user", "content": task_message}
    ]

    print(f"\n=== AI Agent Starting ===")

    while True:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=1000,
        )

        ai_message = response.choices[0].message

        messages.append({
            "role": "assistant",
            "content": ai_message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                }
                for tc in (ai_message.tool_calls or [])
            ] or None
        })

        if ai_message.tool_calls:
            for tool_call in ai_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                tool_result = call_tool(tool_name, tool_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })
        else:
            final_answer = ai_message.content
            print(f"\n=== AI Agent Done ===")
            print(f"Result: {final_answer}")
            print(f"=====================\n")
            return final_answer


def run_assign_agent(patient_id: int, ward: str):
    """
    Runs agent to assign a bed.
    """
    task_message = f"""
You are a hospital bed assignment agent. Your job:
1. Check if there is an empty bed in the {ward} ward.
2. Find which nurse is on duty in that ward.
3. Assign patient {patient_id} to the empty bed with that nurse.
4. Report what you did in simple plain text. Do not use markdown tables.

Use the tools available to you. Do not guess. Use real data.
"""
    return execute_agent_loop(task_message)


def run_discharge_agent(patient_id: int, bed_id: int):
    """
    Runs agent to discharge a patient.
    """
    task_message = f"""
You are a hospital discharge agent. Your job:
1. Discharge patient {patient_id} from bed {bed_id}.
2. Mark the bed as empty/unassigned and log the discharge record.
3. Confirm whether the discharge was successful in simple text.

Use the discharge_patient tool available to you. Do not guess.
"""
    return execute_agent_loop(task_message)


# ============================================================
# STEP 4: Command Line Interface
# ============================================================

if __name__ == "__main__":
    print("\n--- Hospital Bed Management Agent ---")
    print("Make sure FastAPI is running first: uvicorn main:app --reload\n")

    print("Choose action:")
    print("1. Assign Bed")
    print("2. Discharge Patient")
    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        p_id = int(input("Enter patient ID: "))
        w = input("Enter ward (General / Emergency / ICU / Pediatric / Maternity): ").strip()
        run_assign_agent(p_id, w)
    elif choice == "2":
        p_id = int(input("Enter patient ID to discharge: "))
        b_id = int(input("Enter bed ID to free up: "))
        run_discharge_agent(p_id, b_id)
    else:
        print("Invalid choice selected.")