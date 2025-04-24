# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os

from autogen import ConversableAgent

from semantic_kernel.agents.autogen.autogen_conversable_agent import AutoGenConversableAgent

"""
The following sample demonstrates how to use the AutoGenConversableAgent to create a conversation between two agents
where one agent suggests a joke and the other agent generates a joke.

The sample follows the AutoGen flow outlined here:
https://microsoft.github.io/autogen/0.2/docs/tutorial/introduction#roles-and-conversations
"""
from dotenv import load_dotenv

load_dotenv("./.env_console", override=True)

endpoint = os.getenv("AZURE_AI_AGENT_ENDPOINT")
api_key = os.getenv("AZURE_API_KEY")
api_version = os.getenv("AZURE_API_VERSION")
deployment = os.getenv("AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME")


config_list_azure = [
    {
        "api_type": "azure",
        "api_key": api_key,
        "api_version": api_version,
        "base_url": endpoint,  # Use base_url instead of azure_endpoint
        "model": deployment,   # Model name should be at this level
    }
]

async def main():
    cathy = ConversableAgent(
        "cathy",
        system_message="Your name is Cathy and you are a part of a duo of comedians.",
        llm_config={
            "config_list": config_list_azure
        },
        human_input_mode="NEVER",  # Never ask for human input.
    )

    cathy_autogen_agent = AutoGenConversableAgent(conversable_agent=cathy)

    joe = ConversableAgent(
        "joe",
        system_message="Your name is Joe and you are a part of a duo of comedians.",
        llm_config={
            "config_list": config_list_azure
        },
        human_input_mode="NEVER",  # Never ask for human input.
    )

    joe_autogen_agent = AutoGenConversableAgent(conversable_agent=joe)

    async for content in cathy_autogen_agent.invoke(
        recipient=joe_autogen_agent, message="Tell me a joke about the stock market.", max_turns=3
    ):
        print(f"# {content.role} - {content.name or '*'}: '{content.content}'")


if __name__ == "__main__":
    asyncio.run(main())