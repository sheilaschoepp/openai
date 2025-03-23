from openai import OpenAI

class APIHandler:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        f1 = open('system_prompt_sensor_e.txt', 'r')
        self.system_content1 = f1.read()
        f1.close()

    def get_function_from_gpt4(self, prompt):
        messages = []
        # Merge system content with the prompt
        merged_prompt = f"{self.system_content1}\n\n{prompt}"
        user_content = {
            "role": "user",
            "content": merged_prompt
        }
        messages.append(user_content)
        response = self.client.chat.completions.create(
            model="o1-preview-2024-09-12",
            messages=messages
        )
        return response.choices[0].message.content.strip()
    
    def get_function_from_gpt3(self, prompt):
        messages = []
        system_content = {
            "role": "system",
            "content": self.system_content1
        }
        user_content = {
            "role": "user",
            "content": prompt
        }
        messages.append(system_content)
        messages.append(user_content)
        response = self.client.chat.completions.create(
            model = "chatgpt-4o-latest",
            messages = messages
        )
        return response.choices[0].message.content.strip()
