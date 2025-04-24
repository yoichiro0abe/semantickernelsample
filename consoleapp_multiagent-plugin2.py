# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os
import traceback
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.functions import KernelArguments
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.agents import ChatHistoryAgentThread
from dotenv import load_dotenv

load_dotenv("./.env_console_chatagent", override=True)
os.environ.setdefault("OPENAI_TIMEOUT_SECONDS", "30")
# os.environ["OPENAI_TIMEOUT_SECONDS"] = "30" 

# 各エージェントの設定
HAIKU_CREATOR_NAME = "HaikuCreator"
HAIKU_CREATOR_INSTRUCTIONS = (
    "あなたはHaiku Creatorです。ユーザからもらったテーマを元に俳句を作成してください。"
    "俳句は5-7-5の音数で作成してください。"
)

HAIKU_CRITIC_NAME = "HaikuCritic"
HAIKU_CRITIC_INSTRUCTIONS = (
    "あなたはHaiku Criticです。入力として渡された俳句を評価します。"
    "俳句を元に、俳句の良い点と悪い点を説明してください。"
)

COORDINATOR_NAME = "Coordinator"
COORDINATOR_INSTRUCTIONS = (
    "あなたはCoordinatorです。"
    "ユーザからの入力が俳句に場合、HaikuCriticに渡して評価してもらってください。"
    "ユーザの入力が俳句ではない場合、テーマを抽出しHaikuCreatorに渡してください。"
    "HaikuCreatorは、テーマを元に俳句を作成します。"
    "HaikuCreatorが作成した俳句をHaikuCriticに渡してください。"
    "HaikuCriticは、俳句を評価します。"
    "最後に、俳句とその評価をまとめて出力してください。"
)

TASK = "カエル"

# 各エージェントをテストする関数
async def test_agent(agent, prompt, agent_name):
    """単一のエージェントをテストし、結果を表示する関数"""
    print(f"\n----- {agent_name}のテスト開始 -----")
    try:
        print(f"プロンプト: '{prompt}'")
        response = await agent.get_response(messages=prompt)
        print(f"応答: '{response.content}'")
        return response
    except Exception as e:
        print(f"エラー発生: {e}")
        traceback.print_exc()
        return None

# マルチエージェントをテストする関数
async def test_multi_agent(facilitator, prompt):
    """マルチエージェントシステムをテストする関数"""
    print("\n----- マルチエージェントテスト開始 -----")
    try:
        print(f"プロンプト: '{prompt}'")
        response = await facilitator.get_response(messages=prompt)
        print(f"応答: '{response.content}'")
        return response
    except Exception as e:
        print(f"エラー発生: {e}")
        traceback.print_exc()
        return None

# インタラクティブチャット関数
async def interactive_chat(facilitator):
    """ユーザーとエージェントのインタラクティブチャットを実行する関数"""
    thread = None
    print("\n----- インタラクティブモード開始 -----")
    print("終了するには 'exit' または 'quit' と入力するか、Ctrl+Cを押してください")
    
    while True:
        try:
            user_input = input("\nUser:> ")
            if user_input.lower().strip() in ["exit", "quit"]:
                print("\nExiting chat...")
                break
            
            async for response in facilitator.invoke(
                messages=user_input,
                thread=thread,
            ):
 
                print(f"Agent:> {response.name}:{response.content}")
            # if response:
            #     print(f"Agent:> {response.name}:{response.content}")
            #     thread = response.thread
            
        except KeyboardInterrupt:
            print("\nExiting chat...")
            break
        except Exception as e:
            print(f"エラー発生: {e}")
            traceback.print_exc()

async def main():
    # カーネルとサービスのセットアップ
    try:
        service_id = "agent_chat_service"
        kernel = Kernel()
    
        # 環境変数の取得
        deployment_name = os.environ.get("AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME")
        api_key = os.environ.get("AZURE_API_KEY")
        endpoint = os.environ.get("AZURE_AI_AGENT_ENDPOINT")
        api_version = os.environ.get("AZURE_API_VERSION")
    
        if not all([deployment_name, api_key, endpoint, api_version]):
            raise ValueError("Missing one or more Azure OpenAI environment variables.")
    
        print(f"deployment_name: {deployment_name}")
        print(f"endpoint: {endpoint}")
        print(f"api_version: {api_version}")
    
        # チャット完了サービスの設定
        chat_completion = AzureChatCompletion(
            deployment_name=deployment_name,
            api_key=api_key,
            base_url=endpoint,
            api_version=api_version,
            service_id=service_id,
        )
    
        kernel.add_service(chat_completion)
        settings = kernel.get_prompt_execution_settings_from_service_id(service_id=service_id)
        settings.function_choice_behavior = FunctionChoiceBehavior.NoneInvoke()
        print("AzureChatCompletion service added successfully.")
    
        # 個別エージェント作成
        haiku_creator = ChatCompletionAgent(
            kernel=kernel,
            name=HAIKU_CREATOR_NAME,
            instructions=HAIKU_CREATOR_INSTRUCTIONS,
            arguments=KernelArguments(settings=settings),
        )
        
        haiku_critic = ChatCompletionAgent(
            kernel=kernel,
            name=HAIKU_CRITIC_NAME,
            instructions=HAIKU_CRITIC_INSTRUCTIONS,
            arguments=KernelArguments(settings=settings)
        )
        settings2 = kernel.get_prompt_execution_settings_from_service_id(service_id=service_id)
        settings2.function_choice_behavior = FunctionChoiceBehavior.Auto()
        kernel.add_plugin(haiku_creator, plugin_name=HAIKU_CREATOR_NAME)
        kernel.add_plugin(haiku_critic, plugin_name=HAIKU_CRITIC_NAME)
        # ファシリテーターエージェント作成（プラグインとして他のエージェントを使用）
        facilitator = ChatCompletionAgent(
            kernel=kernel,
            name=COORDINATOR_NAME,
            instructions=COORDINATOR_INSTRUCTIONS,
            arguments=KernelArguments(settings=settings2),
            # plugins=[haiku_creator, haiku_critic]
        )
        
        print("All agents created successfully.")

        creator_result = await test_agent(haiku_creator, "春の花", "HaikuCreator")
        print(f"result: {creator_result}")
        if creator_result:
            print("HaikuCreatorのテスト成功")
        else:
            print("HaikuCreatorのテスト失敗")

        critic_result = await test_agent(haiku_critic, "「春の花 香り広がる 青空に」というタイトルで春をテーマにした俳句を評価してください", "HaikuCritic")
        print(f"result: {critic_result}")
        if critic_result:
            print("HaikuCriticのテスト成功")
        else:
            print("HaikuCriticのテスト失敗")

        # すべての個別テストが成功したら、マルチエージェントをテスト
        if creator_result and critic_result:
            multi_result = await test_multi_agent(facilitator, TASK)
        else:
            print("\n個別のエージェントテストに失敗したため、マルチエージェントテストはスキップします")
            multi_result = False
        
        # インタラクティブモードのオプション
        if multi_result:
            try_interactive = input("\nインタラクティブモードを試しますか？(y/n): ")
            if try_interactive.lower() == 'y':
                await interactive_chat(facilitator)
        
    except Exception as e:
        print(f"エラー発生: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

