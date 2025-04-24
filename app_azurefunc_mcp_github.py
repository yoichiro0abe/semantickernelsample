# github のMCPサーバーにアクセスするMCPクライアント

import os
import json
import httpx
import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator
import semantic_kernel as sk
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.text_completion_client_base import TextCompletionClientBase
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.text_content import TextContent
from semantic_kernel.services.ai_service_selector import AIServiceSelector
from semantic_kernel.functions.kernel_function_metadata import KernelFunctionMetadata
from pydantic import Field
from dotenv import load_dotenv
from datetime import datetime
import os

load_dotenv("./.env_azurefunc_mcp_github", override=True)

class GitHubMCPClient(ChatCompletionClientBase, TextCompletionClientBase):
    """GitHub Machine Completion Protocol クライアント (Semantic Kernel と統合)"""
    
    # クラスレベルでのフィールド定義（Pydantic要件）
    base_url: str = "https://api.github.com/mcp/v1"
    auth_token: str = None
    headers: Dict[str, str] = None
    
    def __init__(self, auth_token: str = None):
        """
        GitHub MCP クライアントの初期化
        
        Args:
            auth_token: GitHub Personal Access Token
        """
        # 親クラスを初期化
        ChatCompletionClientBase.__init__(self, ai_model_id="github-copilot")
        TextCompletionClientBase.__init__(self, ai_model_id="github-copilot")
        
        # 属性値を設定
        token = auth_token or os.environ.get("GITHUB_TOKEN")
        if not token:
            raise ValueError("GitHub token is required.")
            
        # Pydanticモデルフィールドを設定
        self.auth_token = token
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }

    async def get_chat_message_contents(
        self, 
        chat_history: ChatHistory, 
        settings: PromptExecutionSettings,
        **kwargs
    ) -> AsyncGenerator[ChatMessageContent, None]:
        """
        チャット履歴に基づいてAIからの応答を非同期的に生成
        
        Args:
            chat_history: チャットの履歴
            settings: プロンプト実行設定
        """
        # チャット履歴からプロンプトを構築
        prompt = ""
        for message in chat_history.messages:
            role = "user" if message.role.lower() == "user" else "assistant"
            prompt += f"{role}: {message.content}\n"
        
        prompt += "assistant: "
        
        # MCPサーバーにリクエスト
        temperature = settings.temperature if settings.temperature is not None else 0.7
        top_p = settings.top_p if settings.top_p is not None else 1.0
        max_tokens = settings.max_tokens if settings.max_tokens is not None else 1000
        
        try:
            # 実際のMCPリクエスト
            response = await self._create_completion(
                prompt=prompt,
                model=self.ai_model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            content = response.get("choices", [{}])[0].get("text", "")
            
            # レスポンスを作成
            result = ChatMessageContent(
                role="assistant",
                content=content,
                ai_model_id=self.ai_model_id
            )
            
            yield result
            
        except Exception as e:
            import traceback
            print(f"Error: {e}")
            traceback.print_exc()  # スタックトレースを表示
            error_message = f"Error in GitHub MCP client: {str(e)}"
            error_content = ChatMessageContent(
                role="assistant",
                content=error_message,
                ai_model_id=self.ai_model_id
            )
            yield error_content
    
    async def complete(
        self, 
        prompt: str, 
        settings: PromptExecutionSettings,
        **kwargs
    ) -> AsyncGenerator[TextContent, None]:
        """
        テキスト補完を生成
        
        Args:
            prompt: 入力プロンプト
            settings: プロンプト実行設定
        """
        temperature = settings.temperature if settings.temperature is not None else 0.7
        top_p = settings.top_p if settings.top_p is not None else 1.0
        max_tokens = settings.max_tokens if settings.max_tokens is not None else 1000
        
        try:
            response = await self._create_completion(
                prompt=prompt,
                model=self.ai_model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            content = response.get("choices", [{}])[0].get("text", "")
            
            result = TextContent(
                text=content,
                ai_model_id=self.ai_model_id
            )
            
            yield result
            
        except Exception as e:
            import traceback
            print(f"Error: {e}")
            traceback.print_exc()  # スタックトレースを表示
            error_message = f"Error in GitHub MCP client: {str(e)}"
            error_content = TextContent(
                text=error_message,
                ai_model_id=self.ai_model_id
            )
            yield error_content
    
    async def _create_completion(self, 
                          prompt: str, 
                          model: str = "copilot", 
                          max_tokens: int = 1000, 
                          temperature: float = 0.7,
                          top_p: float = 1.0,
                          stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        MCP サーバーにテキスト生成リクエストを送信
        
        Args:
            prompt: 入力プロンプト
            model: 使用するモデル
            max_tokens: 生成する最大トークン数
            temperature: 生成の多様性 (0.0〜1.0)
            top_p: 確率分布の上位割合
            stop: 生成を停止する文字列のリスト
            
        Returns:
            サーバーからのレスポンス
        """
        payload = {
            "prompt": prompt,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        if stop:
            payload["stop"] = stop
            
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/completions",
                headers=self.headers,
                json=payload,
                timeout=60.0
            )
            
            if response.status_code != 200:
                raise Exception(f"Error from MCP server: {response.status_code} - {response.text}")
                
            return response.json()
    
    async def get_models(self) -> List[Dict[str, Any]]:
        """利用可能なモデル一覧を取得"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/models",
                headers=self.headers
            )
            
            if response.status_code != 200:
                raise Exception(f"Error fetching models: {response.status_code} - {response.text}")
                
            return response.json()


# Semantic Kernelと統合して使用する例
async def main():
    """基本的な使用例"""
    try:
        # GitHub MCP クライアントを初期化
        mcp_client = GitHubMCPClient()
        
        # Semantic Kernel を初期化
        kernel = sk.Kernel()
        
        # AI サービスを登録
        mcp_client.service_id = "github-copilot"  # 明示的に設定する場合
        kernel.add_service(service=mcp_client)
        
        # チャット履歴の作成
        chat_history = ChatHistory()
        chat_history.add_user_message("Write a Python function to calculate the Fibonacci sequence")
        
        # チャット完了機能を呼び出し
        settings = PromptExecutionSettings(
            service_id="github-copilot",
            temperature=0.7,
            max_tokens=500
        )
        
        # await は削除して、AsyncGenerator を直接取得
        result = kernel.invoke_stream(
            chat_history, 
            settings=settings
        )
        
        # async for で反復処理
        async for chunk in result:
            print(chunk.content, end="", flush=True)
        print("\n")
        
        # テキスト完了の例
        prompt = "Write a function that sorts an array using quicksort:"
        
        # await は削除して、AsyncGenerator を直接取得
        text_result = kernel.invoke_stream(
            prompt, 
            settings=settings
        )
        
        print("\nText completion result:")
        # async for で反復処理
        async for chunk in text_result:
            print(chunk.text, end="", flush=True)
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()  # スタックトレースを表示


if __name__ == "__main__":
    asyncio.run(main())
