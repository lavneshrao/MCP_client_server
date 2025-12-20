import requests
import sys

# Configuration
API_URL = "http://127.0.0.1:8080/chat"
SESSION_ID = "terminal_test_user_01"

def main():
    print(f"--- 🤖 NBFC AI Agent CLI (Session: {SESSION_ID}) ---")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            # 1. Get User Input
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting...")
                break
            
            # 2. Send to API
            payload = {
                "session_id": SESSION_ID,
                "message": user_input
            }
            
            # Print a small loading indicator
            print("...", end="\r") 
            
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            
            # 3. Parse Response
            data = response.json()
            bot_text = data.get("response", "")
            stage = data.get("debug_stage", "UNKNOWN")
            
            # 4. Print Output
            # We explicitly clear the loading line
            print(f"Bot ({stage}): {bot_text}\n")

        except requests.exceptions.ConnectionError:
            print("\n❌ Error: Could not connect to API. Is 'api.py' running?")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()