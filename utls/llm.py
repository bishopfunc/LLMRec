import os
from dataclasses import dataclass
from typing import Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI


@dataclass
class _LLMConfig:
    """LLM 呼び出し関連設定。

    endpoint: Chat/Completion 互換のエンドポイント URL
    model: 利用モデル名
    api_key_env: API キーを格納した環境変数名
    max_retries: 失敗時リトライ最大回数
    retry_sleep: リトライ間隔秒
    timeout: HTTP タイムアウト秒
    """

    endpoint: str = "https://api.openai.com/v1/completions"
    model: str = "gpt-4o"
    api_key_env: str = "OPENAI_API_KEY"  # 無い場合はヘッダに Authorization を付与しない
    max_retries: int = 5
    retry_sleep: float = 8.0
    timeout: float = 60.0

    def build_headers(self) -> Dict[str, str]:
        load_dotenv()
        key = os.getenv(self.api_key_env)
        if key:
            return {"Authorization": f"Bearer {key}"}
        # API Key 無しで動作させる場合 (社内ネットワーク限定など)
        return {}


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
