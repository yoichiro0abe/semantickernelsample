# Copyright (c) Microsoft. All rights reserved.

import asyncio
from typing import Annotated

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion
from semantic_kernel.connectors.mcp import MCPSsePlugin
from semantic_kernel.functions import KernelArguments, kernel_function
from dotenv import load_dotenv
from datetime import datetime
import os

load_dotenv("./.env_console_chatagent", override=True)



# Define a sample plugin for the sample
class MenuPlugin:
    """A sample Menu Plugin used for the concept sample."""

    @kernel_function(description="Provides a list of specials from the menu.")
    def get_specials(self) -> Annotated[str, "Returns the specials from the menu."]:
        return """
        Special Soup: Clam Chowder
        Special Salad: Cobb Salad
        Special Drink: Chai Tea
        """

    @kernel_function(description="Provides the price of the requested menu item.")
    def get_item_price(
        self, menu_item: Annotated[str, "The name of the menu item."]
    ) -> Annotated[str, "Returns the price of the menu item."]:
        # create dict of menu items and their prices
        menu_prices = {
            "clam chowder": "$9.99",
            "cobb salad": "$12.99",
            "chai tea": "$4.99",
        }
        # retrieve the price of the requested menu item
        item_price = menu_prices.get(menu_item.lower(), "$10.00")
        return item_price


# Simulate a conversation with the agent
USER_INPUTS = [
    "Hello",
    "What is the special salad?",
    "What does that cost?",
    "Thank you",
]


async def main():
    # 1. Create the instance of the Kernel to register the plugin and service
    service_id = "agent"
    kernel = Kernel()
    kernel.add_plugin(MenuPlugin(), plugin_name="menu")
    chat_completion = OllamaChatCompletion(
        host="http://localhost:11434",
        ai_model_id="phi4-mini",
        service_id=service_id,
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

    for user_input in USER_INPUTS:
        print(f"# User: {user_input}")
        # 5. Invoke the agent for a response
        async for response in agent.invoke(messages=user_input, thread=thread):
            print(f"# {response.name}: {response}")
            thread = response.thread

    # 6. Cleanup: Clear the thread
    await thread.delete() if thread else None

    """
    Sample output:
    # User: Hello
    # Host: Hello! How can I assist you today?
    # User: What is the special soup?
    # Host: The special soup is Clam Chowder.
    # User: What does that cost?
    # Host: The special soup, Clam Chowder, costs $9.99.
    # User: Thank you
    # Host: You're welcome! If you have any more questions, feel free to ask. Enjoy your day!
    """


if __name__ == "__main__":
    asyncio.run(main())