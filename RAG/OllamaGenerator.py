from .ApiManager import ask_generator_api_call


class OllamaGenerator:

    def __init__(self, model_name: str, system_prompt: str):
        self.model_name = model_name
        self.system_prompt = system_prompt


    def ask_generator(self, user_prompt: str):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = ask_generator_api_call(model_name=self.model_name, messages=messages)
        return response