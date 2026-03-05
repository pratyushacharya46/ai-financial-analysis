import os
import requests
from dotenv import load_dotenv

def load_credentials():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("api key missing")
    return api_key

def test_openrouter_connection(api_key):
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "google/gemma-3-4b-it:free",
        "messages": [
            {
                "role": "user", 
                "content": "Hello! This is a test connection. Please respond with a short greeting."
            }
        ]
    }

    try:
        print("Sending test request to OpenRouter API...")
        response = requests.post(url, headers=headers, json=payload)
        
        # Raise an exception for HTTP error codes
        response.raise_for_status() 
        
        # Parse the JSON response
        response_data = response.json()
        reply = response_data['choices'][0]['message']['content']
        
        print("\nSuccessful!")
        print(f"Model Reply: {reply}")
        
    except requests.exceptions.RequestException as e:
        print("\nFailed!")
        print(f"Error Details: {e}")
        if response is not None and response.text:
            print(f"API Response: {response.text}")

if __name__ == "__main__":
    try:
        key = load_credentials()
        test_openrouter_connection(key)
    except Exception as error:
        print(f"Setup Error: {error}")