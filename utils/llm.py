from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI


class LLMClient:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI()
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
