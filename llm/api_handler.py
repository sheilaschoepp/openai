from openai import OpenAI

class APIHandler:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.messages = []
        f = open('system_prompt.txt', 'r')
        self.system_content = f.read()
        f.close()

    def get_function_from_gpt4(self, prompt):
        system_content = {
            "role": "system",
            "content": self.system_content
        }
        user_content = {
            "role": "user",
            "content": prompt
        }
        self.messages.append(system_content)
        self.messages.append(user_content)
        response = self.client.chat.completions.create(
            model = "gpt-4o",
            messages = self.messages
        )
        return response.choices[0].message.content.strip()
    
    def get_function_from_gpt3(self, prompt):
        system_content = {
            "role": "system",
            "content": self.system_content
        }
        user_content = {
            "role": "user",
            "content": prompt
        }
        self.messages.append(system_content)
        self.messages.append(user_content)
        response = self.client.chat.completions.create(
            model = "gpt-3.5-turbo",
            messages = self.messages
        )
        return response.choices[0].message.content.strip()
