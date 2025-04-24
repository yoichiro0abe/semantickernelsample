import aiohttp
from semantic_kernel.agents import Agent
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from collections.abc import AsyncIterable, Iterable
from openai import AzureOpenAI
import asyncio
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv("./.env_console-o3mini", override=True)

openai_client = AzureOpenAI(
    azure_deployment=os.environ.get("AZURE_DEPLOYMENT_NAME"),
    api_key=os.environ.get("AZURE_API_KEY"),
    azure_endpoint=os.environ.get("AZURE_ENDPOINT"),
    api_version=os.environ.get("AZURE_API_VERSION"),
)

class WebAPIAgent(Agent):
    def __init__(self, name: str, instructions: str, api_url: str):
        super().__init__(id=f"{name}-{hash(name)}", name=name, description=instructions)
        # カスタムプロパティをメタデータに保存
        self._api_url = api_url
        
    @property
    def api_url(self):
        return self._api_url
        
    async def invoke(self, input: str, **kwargs) -> AsyncIterable[ChatMessageContent]:
        api_result = None
        # Web API を呼び出し
        async with aiohttp.ClientSession() as session:
            async with session.get(self.api_url) as api_response:
                api_result = await api_response.text()

        response = openai_client.chat.completions.create(
            model=os.environ.get("AZURE_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": self.description},  # instructionsをdescriptionに保存
                {"role": "user", "content": f"以下のAPIレスポンスから天気予報を作成してください: {api_result}"}
            ]
        )
        llm_content = response.choices[0].message.content

        # LLM と API の結果を結合
        final_content = f"{llm_content}"  # APIの生JSONは非表示にします
        yield ChatMessageContent(role=AuthorRole.ASSISTANT, name=self.name, content=final_content)
    
    async def get_response(self, chat_history, **kwargs):
        # 最後のユーザーメッセージを取得
        user_message = next((msg for msg in reversed(chat_history) if msg.role == AuthorRole.USER), None)
        if user_message:
            async for response in self.invoke(user_message.content, **kwargs):
                return response
        return ChatMessageContent(role=AuthorRole.ASSISTANT, name=self.name, content="申し訳ありませんが、ユーザーメッセージがありません。")
    
    async def invoke_stream(self, input: str, **kwargs) -> AsyncIterable[ChatMessageContent]:
        # Web API を呼び出し
        async with aiohttp.ClientSession() as session:
            async with session.get(self.api_url) as api_response:
                api_result = await api_response.text()

        # ストリーミングでレスポンスを取得 (awaitを削除)
        stream = openai_client.chat.completions.create(
            model=os.environ.get("AZURE_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": self.description},
                {"role": "user", "content": f"以下のAPIレスポンスから天気予報を作成してください: {api_result}"}
            ],
            stream=True  # ストリーミングを有効化
        )

        accumulated_content = ""
        # async forではなく通常のforを使用
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                if chunk.choices[0].delta.content:
                    content_delta = chunk.choices[0].delta.content
                    accumulated_content += content_delta
                    # 少しずつ内容を返す
                    yield ChatMessageContent(
                        role=AuthorRole.ASSISTANT,
                        name=self.name,
                        content=accumulated_content
                    )
                    # ストリーミング感を出すために少し待機
                    # await asyncio.sleep(0.05)

# 使用例
agent = WebAPIAgent(
    name="Reporter",
    instructions="あなたはJSONを解釈できるアナウンサーです。与えられたJSONからいい感じに天気予報を作成してください。",
    api_url="https://www.jma.go.jp/bosai/forecast/data/forecast/130000.json"
)

async def test_agent():
    # invoke をテスト
    starttime = datetime.now()
    print(f"\n--- Testing invoke method ---{starttime.strftime('%Y-%m-%d %H:%M:%S')}")
    async for content in agent.invoke("今日の東京の朝の天気教えて"):
        print(f"# {content.role} - {content.name}: '{content.content}'")
    endtime = datetime.now()
    print(f"\n--- End of invoke ---{endtime.strftime('%Y-%m-%d %H:%M:%S')}")
    # invoke_stream をテスト
    starttime = datetime.now()
    print(f"\n--- Testing invoke_stream method  ---{starttime.strftime('%Y-%m-%d %H:%M:%S')}")
    previous_length = 0
    async for content in agent.invoke_stream("明日の東京の夜の天気を教えて"):
        current_content = content.content
        # 新しい部分だけを表示
        new_content = current_content[previous_length:]
        if new_content:
            print(f"{new_content}", end="", flush=True)
            previous_length = len(current_content)
    # end timeをprint
    endtime = datetime.now()
    print(f"\n--- End of streaming ---{endtime.strftime('%Y-%m-%d %H:%M:%S')}")


# 修正したテスト関数を実行
asyncio.run(test_agent())