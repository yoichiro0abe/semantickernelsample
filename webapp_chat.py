import asyncio
from typing import Annotated, AsyncGenerator
import os
import json
import uuid
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState # starlette.websocketsからインポート
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
# Jinja2Templatesをインポート
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, AzureChatCompletion
# AzureTextEmbedding is removed as it's not used with VolatileMemoryStore directly here
# from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.exceptions import ContentInitializationError, ContentSerializationError

# Correct import for VolatileMemoryStore
from semantic_kernel.memory.volatile_memory_store import VolatileMemoryStore
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole

# MemoryRecordのインポートを追加 (履歴保存に必要)
from semantic_kernel.memory.memory_record import MemoryRecord
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.functions import KernelArguments, kernel_function
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
import uvicorn
# Import SessionMiddleware
from starlette.middleware.sessions import SessionMiddleware

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


# Load environment variables
load_dotenv("./.env_console_chatagent", override=True)

# Initialize FastAPI app
app = FastAPI()

# --- Jinja2Templatesの設定 ---
# templatesディレクトリを指定
templates = Jinja2Templates(directory="templates")

# Add Session Middleware - Make sure to set a secure secret key in production
# Replace with a real secret key from env or config
app.add_middleware(SessionMiddleware, secret_key=os.environ.get("SESSION_SECRET_KEY", "your-super-secret-key-here"))

# Global variables
agent = None
volatile_store = None # Initialize volatile_store globally
connections = {} # Store WebSocket connections and their associated threads/history

# Setup memory store (using VolatileMemoryStore)
async def setup_memory():
    global volatile_store
    # Use VolatileMemoryStore (no embeddings needed for this simple history)
    volatile_store = VolatileMemoryStore()
    # Create the collection explicitly if needed, though upsert might handle it
    # Depending on the exact VolatileMemoryStore implementation, explicit creation might be safer.
    # await volatile_store.create_collection("chathistory") # Volatile doesn't need explicit creation usually
    # No separate 'memory' object needed when just using the store for history
    return volatile_store, None

async def save_chat_history(conversationid: str, chat_history: ChatHistory):
    if not volatile_store:
        print("Error: Volatile store not initialized.")
        return
    try:
        # ChatHistoryオブジェクトをJSON文字列にシリアライズ
        chat_data_str = chat_history.serialize() # または chat_history.model_dump_json()

        # VolatileMemoryStoreに保存
        if "chathistory" not in volatile_store._store:
             volatile_store._store["chathistory"] = {}
        volatile_store._store["chathistory"][conversationid] = chat_data_str
        print(f"History saved for {conversationid} (serialized)")

    except ContentSerializationError as e:
        print(f"Error serializing history for {conversationid}: {e}")
    except Exception as e:
        print(f"Error saving history for {conversationid}: {e}")


async def get_chat_history(conversationid: str) -> ChatHistory:
    if not volatile_store:
        print("Error: Volatile store not initialized.")
        return ChatHistory()
    try:
        if "chathistory" in volatile_store._store and conversationid in volatile_store._store["chathistory"]:
            chat_data_str = volatile_store._store["chathistory"][conversationid]
            if chat_data_str:
                # JSON文字列からChatHistoryオブジェクトを復元
                chat_history = ChatHistory.restore_chat_history(chat_data_str) # または ChatHistory.model_validate_json(chat_data_str)
                print(f"History retrieved for {conversationid} (deserialized), length: {len(chat_history)}")
                return chat_history
        else:
            print(f"No history found for {conversationid}")

    except ContentInitializationError as e:
         print(f"Error deserializing history for {conversationid}: {e}")
         # 不正な形式の場合は新しい履歴を返す
         pass
    except Exception as e:
        print(f"Error retrieving history for {conversationid}: {e}")
        pass
    return ChatHistory()
# Streaming chat response function (modified for /chat endpoint using agent.invoke_stream)
async def stream_chat_response(conversationid: str, user_input: str) -> AsyncGenerator[str, None]:
    # Retrieve history associated with the conversation ID
    chat_history = await get_chat_history(conversationid)
    chat_history.add_user_message(user_input)

    # Use the global agent
    global agent
    if not agent:
        yield "Error: Agent not initialized."
        return

    full_response = ""
    try:
        # Use agent.invoke_stream with the full history in messages
        # The agent should handle context and function calling based on this
        async for response_chunk in agent.invoke_stream(
            messages=chat_history.messages, # Pass the list of messages
            # thread=None, # Explicitly not using a persistent thread object here
            # arguments=... # Usually not needed if agent is initialized with settings
        ):
            # Process the streaming content (assuming ChatMessageContent)
            chunk_content = ""
            sender_name = "Host" # Default

            if hasattr(response_chunk, 'content'):
                chunk_content = str(response_chunk.content)
            # Add other checks if needed (like in the original websocket handler)
            elif isinstance(response_chunk, str):
                 chunk_content = response_chunk
            elif hasattr(response_chunk, 'text'):
                 chunk_content = str(response_chunk.text)
            else:
                 chunk_content = str(response_chunk)

            if hasattr(response_chunk, 'name') and response_chunk.name:
                 sender_name = response_chunk.name
            elif hasattr(response_chunk, 'role') and response_chunk.role:
                 sender_name = str(response_chunk.role).capitalize()


            if chunk_content:
                # Yield only the content part for simple text streaming
                yield chunk_content
                full_response += chunk_content

        # Save the updated history (user input + full assistant response)
        if full_response:
            # Ensure the role added matches what ChatHistory expects (e.g., AuthorRole.ASSISTANT or "assistant")
            chat_history.add_assistant_message(full_response)
            await save_chat_history(conversationid, chat_history)
        else:
            print(f"Warning: Empty response received for {conversationid} via HTTP")
            # Optionally save history even with empty response?
            # await save_chat_history(conversationid, chat_history)

    except Exception as e:
        import traceback
        print(f"Error during agent invocation for {conversationid} via HTTP: {e}")
        traceback.print_exc() # Print stack trace for debugging
        yield f"Error: {e}"
        # Optionally save history even on error?
        # chat_history.add_assistant_message(f"Error: {e}")
        # await save_chat_history(conversationid, chat_history)


# Chat endpoint using HTTP Streaming
@app.get("/chat")
async def chat(request: Request, message: str):
    # Get or create a unique conversation ID for the session
    conversationid = request.session.get("conversationid")
    if not conversationid:
        conversationid = str(uuid.uuid4())
        request.session["conversationid"] = conversationid
        print(f"New conversation started (HTTP): {conversationid}")
    else:
        print(f"Continuing conversation (HTTP): {conversationid}")

    # Return a streaming response
    return StreamingResponse(stream_chat_response(conversationid, message), media_type="text/plain; charset=utf-8") # Ensure UTF-8


# Agent setup function
async def setup_agent():
    global agent

    # Kernel setup
    service_id = "agent_chat_service" # Use a distinct service ID if needed
    kernel = Kernel()
    kernel.add_plugin(MenuPlugin(), plugin_name="menu")

    # Configure Azure Chat Completion service
    try:
        # Ensure environment variables are loaded and correct
        deployment_name = os.environ.get("AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME")
        api_key = os.environ.get("AZURE_API_KEY")
        endpoint = os.environ.get("AZURE_AI_AGENT_ENDPOINT") # Use 'endpoint' not 'base_url'
        api_version = os.environ.get("AZURE_API_VERSION")

        if not all([deployment_name, api_key, endpoint, api_version]):
             raise ValueError("Missing one or more Azure OpenAI environment variables.")

        chat_completion = AzureChatCompletion(
            deployment_name=deployment_name,
            api_key=api_key,
            base_url=endpoint,
            api_version=api_version,
            service_id=service_id
        )
        kernel.add_service(chat_completion)
        print("AzureChatCompletion service added successfully.")
    except Exception as e:
        print(f"Error setting up AzureChatCompletion: {e}")
        # Consider raising the exception or handling it appropriately
        # For now, we'll print and continue, but the agent might fail
        return # Stop agent setup if service fails

    # Configure function choice behavior
    settings = kernel.get_prompt_execution_settings_from_service_id(service_id=service_id)
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
    print("Function choice behavior set to Auto.")

    # Create the agent instance
    agent = ChatCompletionAgent(
        kernel=kernel,
        name="Host",
        instructions="Answer questions about the menu. Use the available tools if needed.",
        arguments=KernelArguments(settings=settings),
    )
    print("ChatCompletionAgent created successfully.")


# --- html と html2 変数を削除 ---

# Startup event to initialize memory and agent
@app.on_event("startup")
async def startup_event():
    print("Starting up...")
    await setup_memory()
    print("Memory setup complete.")
    await setup_agent()
    # Agent setup might fail if Azure creds are wrong, check if agent is None
    if agent:
        print("Agent setup complete.")
    else:
        print("Agent setup failed. Check Azure credentials and configuration.")
    print("Startup finished.")

# --- ルートエンドポイントを修正 ---
@app.get("/")
async def get(request: Request): # requestを追加
    # websocket_chat.html をレンダリング
    return templates.TemplateResponse("websocket_chat.html", {"request": request})

# --- /stream エンドポイントを修正 ---
@app.get("/stream")
async def get_stream_page(request: Request): # requestを追加
   # http_stream_chat.html をレンダリング
   return templates.TemplateResponse("http_stream_chat.html", {"request": request})


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = None
    agent_thread = None # Use local variable for thread per connection
    chat_history = ChatHistory() # Keep history per connection

    try:
        while True:
            data_text = await websocket.receive_text()
            data = json.loads(data_text)
            print(f"WS Received from {client_id}: {data}") # Log received data

            if data['type'] == 'init':
                client_id = data['clientId']
                # Store WebSocket and initialize history/thread for this client
                connections[client_id] = {
                    'websocket': websocket,
                    'thread': None, # Thread will be created/managed by agent.invoke
                    'history': ChatHistory() # Start fresh history for WS connection
                }
                print(f"WebSocket client connected: {client_id}")
                await websocket.send_json({'type': 'info', 'message': f'Connected with ID {client_id}'})
                chat_history = connections[client_id]['history'] # Use the history for this connection

            elif data['type'] == 'message':
                if not client_id or client_id not in connections:
                     print(f"WS Error: Client not initialized or not found: {client_id}")
                     await websocket.send_json({'type': 'error', 'message': 'Client not initialized or connection lost.'})
                     continue # Skip processing if client ID is invalid

                user_input = data['message']
                current_connection = connections.get(client_id)

                if not current_connection:
                    print(f"WS Error: Connection info not found for client {client_id}")
                    await websocket.send_json({'type': 'error', 'message': 'Connection info not found.'})
                    continue

                # Add user message to this connection's history
                chat_history.add_user_message(user_input)

                # Send user message back to client for display (handled client-side now)
                # await websocket.send_json({
                #     'type': 'message',
                #     'sender': 'User',
                #     'message': user_input
                # })

                # Invoke agent and stream response
                full_response = ""
                agent_thread = current_connection.get('thread') # Get thread if exists

                if not agent:
                     print(f"WS Error: Agent not initialized for client {client_id}")
                     await websocket.send_json({'type': 'error', 'message': 'Agent not available.'})
                     continue

                try:
                    # Use agent.invoke with the connection's history
                    # The agent should manage the thread internally based on history/messages
                    # Pass the current history explicitly if needed by the agent implementation
                    # Note: ChatCompletionAgent might implicitly use history if passed via messages/thread
                    # Let's rely on the agent managing state via thread, passing only new message
                    async for response_chunk in agent.invoke(messages=user_input, thread=agent_thread):
                        # response_chunk is likely a ChatMessageContent or similar
                        chunk_content = ""
                        sender_name = "Host" # Default sender name

                        if hasattr(response_chunk, 'content'):
                            chunk_content = str(response_chunk.content)
                        elif isinstance(response_chunk, str):
                            chunk_content = response_chunk
                        elif hasattr(response_chunk, 'text'):
                            chunk_content = str(response_chunk.text)
                        else:
                            chunk_content = str(response_chunk)

                        if hasattr(response_chunk, 'name') and response_chunk.name:
                            sender_name = response_chunk.name
                        elif hasattr(response_chunk, 'role') and response_chunk.role:
                            # Map role to name if needed, e.g., AuthorRole.ASSISTANT -> "Host"
                            sender_name = str(response_chunk.role).capitalize()


                        if chunk_content:
                            full_response += chunk_content
                            await websocket.send_json({
                                'type': 'stream_chunk', # Use a specific type for chunks
                                'sender': sender_name,
                                'message': chunk_content
                            })

                        # Update the thread state for this connection if returned by invoke
                        if hasattr(response_chunk, 'thread') and response_chunk.thread:
                             current_connection['thread'] = response_chunk.thread
                             agent_thread = response_chunk.thread # Update local variable too

                    # After streaming, add the full assistant message to history
                    if full_response:
                        chat_history.add_assistant_message(full_response)
                    else:
                         print(f"WS Warning: Empty response for client {client_id}")


                except Exception as e:
                    print(f"Error during agent invocation for {client_id}: {e}")
                    await websocket.send_json({'type': 'error', 'message': f'Agent invocation error: {e}'})

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for client {client_id}")
        if client_id and client_id in connections:
            # Optional: Clean up thread on disconnect if desired
            # thread_to_delete = connections[client_id].get('thread')
            # if thread_to_delete:
            #     try:
            #         # await thread_to_delete.delete() # Agent threads might not have explicit delete
            #         print(f"Cleaned up resources for client {client_id}")
            #     except Exception as e:
            #         print(f"Error cleaning up resources for {client_id}: {e}")
            del connections[client_id] # Remove connection info
    except Exception as e:
        print(f"An error occurred in WebSocket handler for {client_id}: {e}")
        # Attempt to inform the client about the error
        if client_id and client_id in connections and connections[client_id]['websocket'].client_state == WebSocketState.CONNECTED:
             try:
                 await connections[client_id]['websocket'].send_json({'type': 'error', 'message': f'Server error: {e}'})
             except Exception as send_err:
                 print(f"Could not send error to disconnected client {client_id}: {send_err}")
        if client_id and client_id in connections:
            del connections[client_id] # Clean up connection entry on error


if __name__ == "__main__":
    # templatesディレクトリとHTMLファイルが存在しない場合に作成するコードは省略
    # 事前に手動で作成するか、必要であれば追加してください
    print("Starting Uvicorn server...")
    # Ensure reload is False if you're managing global state carefully in production
    uvicorn.run("webapp_chat:app", host="localhost", port=8000, reload=True)
