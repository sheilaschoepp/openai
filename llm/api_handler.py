from openai import OpenAI

class APIHandler:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.messages = []

    def get_function_from_gpt4(self, prompt):
        content = {
            "role": "user",
            "content": prompt
        }
        self.messages.append(content)
        response = self.client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = self.messages,
            temperature=1.3
        )
        return response.choices[0].message.content.strip()
    
    def get_function_from_gpt3(self, prompt):
        content = {
            "role": "user",
            "content": prompt
        }
        self.messages.append(content)
        response = self.client.chat.completions.create(
            model = "gpt-3.5-turbo",
            messages = self.messages,
            temperature=1.3
        )
        return response.choices[0].message.content.strip()
