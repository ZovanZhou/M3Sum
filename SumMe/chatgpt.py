import requests
import json

class ChatGPT(object):
    def __init__(self, api_key: str = "xxxx"):
        self.url = "https://api.openai-proxy.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        self.data = {
          "model": "gpt-3.5-turbo",
          "messages": [],
          "temperature": 0,
          "max_tokens": 2048,
          "frequency_penalty": 0.0,
          "presence_penalty": 1
        }

    def respond(self, user_prompt):
        self.data["messages"] = [
            {"role": "user", "content": user_prompt}
        ]
        response = requests.post(self.url, headers=self.headers, data=json.dumps(self.data))
        try:
            response = json.loads(response.content)
        except Exception as ex:
            print(user_prompt)
            response = {}

        if "choices" in response:
            message = response["choices"][0]["message"]["content"]
        else:
            print(response)
            message = ""
        return message