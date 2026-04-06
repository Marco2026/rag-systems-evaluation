from Config.settings import Settings
import requests

def create_embedding_api_call(model_name: str, input: list[str]) -> dict[str:str]:
        settings = Settings()
        uri = "https://llamus.cs.us.es/ollama/api/embed"
        
        response = requests.post(
            uri,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.llamus_api_key}"
            },
            json={
                "stream": False,
                "model": model_name,
                "input": input
            }
        )
        response.raise_for_status()
        data = response.json()
        embeddings = data["embeddings"]
        return embeddings


def ask_generator_api_call(model_name: str, messages: list[dict[str:str]]) -> dict[str:str]:
        settings = Settings()
        uri = "https://llamus.cs.us.es/ollama/api/chat"
        
        response = requests.post(
            uri,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.llamus_api_key}"
            },
            json={
                "stream": False,
                "model": model_name,
                "messages": messages
            }
        )
        response.raise_for_status()
        data = response.json()
        answer = data["message"]["content"]
        return answer