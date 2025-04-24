# Copyright (c) Microsoft. All rights reserved.

import asyncio
from typing import Annotated

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions import KernelArguments, kernel_function
from dotenv import load_dotenv
from datetime import datetime
import os
from semantic_kernel.connectors.mcp import MCPStdioPlugin, MCPPluginBase

load_dotenv("./.env_console_chatagent", override=True)


async def create_mcp_plugin() -> MCPStdioPlugin:
    
    plugin = MCPStdioPlugin(
        name="example_plugin",
        command="python",
        args=["C:/Users/v-yoiabe/projects/mcp-server-1/server_stdio.py"],
        env={},
        encoding="utf-8",
    )
    await plugin.connect()
    return plugin

async def check_tools_in_mcp(plugin: MCPPluginBase):
    """
    MCPStdioPlugin を用いて、MCPクライアントに接続後
    利用可能なツールの一覧を取得。
    """

    # 3. MCPサーバが提供するツール一覧を取得
    tools_response = await plugin.session.list_tools()
    tools = tools_response.tools if tools_response else []
    print("取得したツール一覧:")
    for t in tools:
        print(f" - {t.name}: {t.description}")

    return tools

async def main():
    diyplugin = await create_mcp_plugin()
    await check_tools_in_mcp(diyplugin)
    # 1. Create the instance of the Kernel to register the plugin and service
    service_id = "agent"
    kernel = Kernel()
    kernel.add_plugin(diyplugin)
    chat_completion = AzureChatCompletion(
        deployment_name=os.environ.get("AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME"),
        api_key=os.environ.get("AZURE_API_KEY"),
        base_url=os.environ.get("AZURE_AI_AGENT_ENDPOINT"),
        api_version=os.environ.get("AZURE_API_VERSION"),
        service_id=service_id
    )
    kernel.add_service(chat_completion)

    # 2. Configure the function choice behavior to auto invoke kernel functions
    # so that the agent can automatically execute the menu plugin functions when needed
    settings = kernel.get_prompt_execution_settings_from_service_id(service_id=service_id)
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    # 3. Create the agent
    agent = ChatCompletionAgent(
        kernel=kernel,
        name="Host",
        instructions="Answer questions about the menu.",
        arguments=KernelArguments(settings=settings),
    )

    # 4. Create a thread to hold the conversation
    # If no thread is provided, a new thread will be
    # created and returned with the initial response
    thread: ChatHistoryAgentThread = None

    # 標準入力を受け取ってループする
    while True:
        user_input = input("# User: ")
        if user_input == "exit":
            break

        # print(f"# User: {user_input}")
        # 5. Invoke the agent for a response
        async for response in agent.invoke(messages=user_input, thread=thread):
            print(f"# {response.name}: {response}")
            thread = response.thread
        # 6. Cleanup: Clear the thread
    await thread.delete() if thread else None


if __name__ == "__main__":
    asyncio.run(main())