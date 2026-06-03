import requests
from rich.console import Console
from rich.markdown import Markdown

PATHWAY_URL = "http://localhost:8011/"

console = Console()
def run_http_client():
    print("RAG Pipeline chat")
    print("Type your query and press Enter. (Type 'exit' to quit)\n")

    while True:
        try:
            user_input = input("You: ")
            
            if user_input.strip().lower() in ['exit' ,]:
                print("Exiting...")
                break
            
            if not user_input.strip():
                continue

            payload = {
                "messages": user_input
            }
            
            response = requests.post(PATHWAY_URL, json=payload)
            
            if response.status_code == 200:
                console.print(Markdown(f"RESPONSE : {response.text.strip()} \n"))
                print("-" * 50)
            else:
                print(f"[Error] Status {response.status_code}: {response.text}\n")

        except requests.exceptions.ConnectionError:
            print(f"[Error] Could not connect to {PATHWAY_URL}.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    run_http_client()