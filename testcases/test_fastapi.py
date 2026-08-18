import asyncio
import json
from unittest.mock import MagicMock
from main import chat
from models import ChatRequest
from auth import AuthenticatedUser
from dotenv import load_dotenv

load_dotenv()

async def run_test():
    print("--- Testing main.py /chat Function Directly ---")
    
    # 1. Mock Request
    req = ChatRequest(
        message="What is the penalty for drunk driving?",
        language="en",
        stream=False
    )
    
    # 2. Mock Token
    token = AuthenticatedUser(
        user_id="test_user_id",
        name="Test User",
        email="test@example.com",
        roles=["User"]
    )
    
    # 3. Mock DB Session
    db = MagicMock()
    mock_session = MagicMock()
    mock_session.id = "mock_session_id"
    db.query.return_value.filter.return_value.first.return_value = mock_session
    
    print(f"Calling chat() directly with message: {req.message}\n")
    
    try:
        response = await chat(req=req, token=token, db=db)
        print("SUCCESS! Function executed successfully.")
        print("\nReturned Dictionary from main.py:")
        print(json.dumps(response, indent=2))
    except Exception as e:
        print(f"FAILED! Error: {e}")

if __name__ == "__main__":
    asyncio.run(run_test())
