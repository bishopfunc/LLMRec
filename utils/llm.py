import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI


class LLMClient:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model_name = "gpt-4o"

    def sample(self, prompt: str) -> Optional[str]:
        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
        )
        return response.output_text


def main():
    client = LLMClient()
    prompt = "Hello, how are you?"
    ouput = client.sample(prompt)
    print(ouput)


if __name__ == "__main__":
    main()
