"""
Here is the problem:
Your boss have a request to AI team:
Our company need a AI agent to process message when everyone is sleeping.
Let's build a agent and get me sleep well :xD

Ok now let's building it.
First, break this problem as context:
- Our company: ecommerce company.
- What is the category of each task? Ok ask my boss.


Ok, to easy testing, input user's message will be prepare as json file.
"""

import json
import os

from pathlib import Path
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# OK now we experiment some opensource model
GPT_OSS_20B: str = "openai/gpt-oss-20b"
GPT_OSS_120B: str = "openai/gpt-oss-120b"
META_LLAMA_31_70B_INSTRUCT: str = "meta/llama-3.1-70b-instruct"

NVIDIA_MODEL: str = META_LLAMA_31_70B_INSTRUCT or os.getenv("NVIDIA_DEFAULT_MODEL", "")


SYSTEM_PROMPT = """You're a e-commerce assistant to classify user's request in one of:
- order status
- returns
- product inquiry
- payments
- account management
- technical
- feedback
- general

Then summary information MUST HAVE these information:
- summary
- key problems
- suggest solution
"""


class BasicUserRequestClassificationAgent:
    """
    Agent will classify email, phone call, direct message from user as a category.
    This idea very simple, using LLM model to classify it (doesn't have structure).
    Just using prompt engineering to guide LLM to do this task.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str,
        system: str = "",
    ) -> None:
        self._client = client
        self._model = model
        self._system = system
        self._messages: List = []
        if self._system:
            self._messages.append({"role": "system", "content": f"{self._system}"})

    def __call__(self, message: str):
        if message:
            self._messages.append({"role": "user", "content": f"{message}"})

        final_assistant_content = self.execute(message)

        if final_assistant_content:
            self._messages.append(
                {"role": "assistant", "content": f"{final_assistant_content}"}
            )

        return final_assistant_content

    def execute(self, user_prompt: str):
        completions = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": f"Here is user's message: '''{user_prompt}'''",
                },
            ],
        )
        response = completions.choices[0].message.content
        print(response)


if __name__ == "__main__":
    # Get user requests data
    root_dir = Path(__file__).parents[2]
    file_path = root_dir / "data/user_requests.json"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # pprint.pprint(data)

    user_messages = data.get("user_requests")

    client = OpenAI(
        base_url=os.getenv("NVIDIA_BASE_URL", ""),
        api_key=os.getenv("NVIDIA_API_KEY", ""),
    )

    agent = BasicUserRequestClassificationAgent(
        client, model=NVIDIA_MODEL, system=SYSTEM_PROMPT
    )

    print(agent)
    # for message in user_messages:
    #     print("===" * 10 + f"\tPROCESSING FOR {message.get('user', '')}\t" + "===" * 10)
    #     agent(message.get("content", ""))
    #     print("===" * 30)
