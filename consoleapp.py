import asyncio
import logging
import os
from semantic_kernel import Kernel
from semantic_kernel.utils.logging import setup_logging
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_arguments import KernelArguments

from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
from plugin import LightsPlugin,WeatherPlugin, CurrentDatePlugin

from dotenv import load_dotenv

load_dotenv("./.env_console", override=True)
# 環境変数の値を確認するコードを追加
print("AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME:", os.environ.get("AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME"))
print("AZURE_API_KEY:", os.environ.get("AZURE_API_KEY")[:5] + "..." if os.environ.get("AZURE_API_KEY") else None)
print("AZURE_AI_AGENT_ENDPOINT:", os.environ.get("AZURE_AI_AGENT_ENDPOINT"))
print("AZURE_API_VERSION:", os.environ.get("AZURE_API_VERSION"))

async def main():
    # Initialize the kernel
    kernel = Kernel()

    # Add Azure OpenAI chat completion
    chat_completion = AzureChatCompletion(
        deployment_name=os.environ.get("AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME"),
        api_key=os.environ.get("AZURE_API_KEY"),
        base_url=os.environ.get("AZURE_AI_AGENT_ENDPOINT"),
        api_version=os.environ.get("AZURE_API_VERSION"),
    )
    kernel.add_service(chat_completion)

    # Set the logging level for  semantic_kernel.kernel to DEBUG.
    setup_logging()
    logging.getLogger("kernel").setLevel(logging.DEBUG)

    # Add a plugin (the LightsPlugin class is defined below)
    kernel.add_plugin(
        LightsPlugin(),
        plugin_name="Lights",
    )
    weather_plugin = WeatherPlugin()
    await weather_plugin.load_area_codes()
    kernel.add_plugin(
        weather_plugin,
        plugin_name="Weather",
    )
    current_plugin = CurrentDatePlugin()
    kernel.add_plugin(
        current_plugin,
        plugin_name="CurrentDate",
    )
    # Enable planning
    execution_settings = AzureChatPromptExecutionSettings()
    execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    # Create a history of the conversation
    history = ChatHistory()

    # Initiate a back-and-forth chat
    userInput = None
    while True:
        # Collect user input
        userInput = input("User > ")

        # Terminate the loop if the user says "exit"
        if userInput == "exit":
            break

        # Add user input to the history
        history.add_user_message(userInput)

        # Get the response from the AI
        result = await chat_completion.get_chat_message_content(
            chat_history=history,
            settings=execution_settings,
            kernel=kernel,
        )

        # Print the results
        print("Assistant > " + str(result))

        # Add the message from the agent to the chat history
        history.add_message(result)

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())