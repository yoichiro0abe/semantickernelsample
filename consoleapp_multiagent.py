import asyncio
import logging
import os
from semantic_kernel import Kernel
from semantic_kernel.utils.logging import setup_logging
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
from dotenv import load_dotenv

load_dotenv("./.env_console", override=True)

# Configure logging
setup_logging()
logging.getLogger("kernel").setLevel(logging.DEBUG)

class SharedPlugin:
    @kernel_function(
        name="get_current_time",
        description="Gets the current time.",
    )
    def get_current_time(self) -> str:
        """Gets the current time."""
        import datetime
        now = datetime.datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S")

class PlannerAgent:
    def __init__(self, kernel: Kernel, name: str, chat_completion: AzureChatCompletion):
        self.kernel = kernel
        self.name = name
        self.history = ChatHistory()
        self.execution_settings = AzureChatPromptExecutionSettings(
            function_choice_behavior=FunctionChoiceBehavior.Auto()
        )
        self.system_message = "You are a planner agent. Your role is to create plans based on user requests."
        self.history.add_system_message(self.system_message)
        self.chat_completion = chat_completion

    async def send_message(self, message: str, recipient: "ExecutorAgent") -> str:
        self.history.add_user_message(message)
        result = await self.chat_completion.get_chat_message_content(
            chat_history=self.history,
            settings=self.execution_settings,
            kernel=self.kernel
        )
        self.history.add_message(result)
        recipient.receive_message(result.content)
        return result.content

class ExecutorAgent:
    def __init__(self, kernel: Kernel, name: str, chat_completion: AzureChatCompletion):
        self.kernel = kernel
        self.name = name
        self.history = ChatHistory()
        self.execution_settings = AzureChatPromptExecutionSettings(
            function_choice_behavior=FunctionChoiceBehavior.Auto()
        )
        self.system_message = "You are an executor agent. Your role is to execute plans provided by the planner agent."
        self.history.add_system_message(self.system_message)
        self.chat_completion = chat_completion

    def receive_message(self, message: str) -> None:
        self.history.add_assistant_message(message)

    async def execute_plan(self) -> str:
        result = await self.chat_completion.get_chat_message_content(
            chat_history=self.history,
            settings=self.execution_settings,
            kernel=self.kernel
        )
        self.history.add_message(result)
        return result.content

async def main():
    # Initialize Azure OpenAI chat completion
    chat_completion = AzureChatCompletion(
        deployment_name=os.environ.get("AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME"),
        api_key=os.environ.get("AZURE_API_KEY"),
        base_url=os.environ.get("AZURE_AI_AGENT_ENDPOINT"),
        api_version=os.environ.get("AZURE_API_VERSION"),
    )

    # Create kernels for each agent
    planner_kernel = Kernel()
    planner_kernel.add_service(chat_completion)
    executor_kernel = Kernel()
    executor_kernel.add_service(chat_completion)

    # Add shared plugin
    shared_plugin = SharedPlugin()
    planner_kernel.add_plugin(shared_plugin, plugin_name="SharedPlugin")
    executor_kernel.add_plugin(shared_plugin, plugin_name="SharedPlugin")

    # Create agents
    planner = PlannerAgent(planner_kernel, "Planner", chat_completion)
    executor = ExecutorAgent(executor_kernel, "Executor", chat_completion)

    # Example conversation
    user_request = "What time is it now?"
    await planner.send_message(user_request, executor)
    result = await executor.execute_plan()
    print(f"Executor > {result}")

    user_request = "What time is it now? and tell me again."
    await planner.send_message(user_request, executor)
    result = await executor.execute_plan()
    print(f"Executor > {result}")

if __name__ == "__main__":
    asyncio.run(main())
