import asyncio
from mcp import ClientSession,stdio_client,StdioServerParameters
from openai import AsyncAzureOpenAI
import os
from dotenv import load_dotenv
from datetime import datetime
import sys

load_dotenv("./.env_azurefunc_mcp_github", override=True)
# Azure OpenAIの設定
azure_endpoint = os.getenv("AZURE_AI_AGENT_ENDPOINT")  # 例: "https://your-resource.openai.azure.com/"
api_key = os.getenv("AZURE_API_KEY")
deployment_name = os.getenv("AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME")  # デプロイ名を指定

# Azure OpenAIクライアントの初期化
client = AsyncAzureOpenAI(
    azure_deployment=deployment_name,
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    api_version=os.getenv("AZURE_API_VERSION")  # 必要に応じてバージョンを調整
)

async def list_mcp_tools():
    env = os.environ.copy()
    env["GITHUB_TOKEN"] = os.getenv("GITHUB_TOKEN", "")
    
    server_params = StdioServerParameters(
        command="npx",
        args=["--yes", "@modelcontextprotocol/server-github"],
        # command="python",
        # args=["mcp_proxy.py"],
        env=env,
        shell=False
    )
    
    try:
        print(f"GITHUB_TOKEN設定: {'設定済み' if env.get('GITHUB_TOKEN') else '未設定'}")
        
        # MCPサーバーにstdioで接続
        async with stdio_client(server_params, errlog=sys.stderr) as transport:
            print(f"MCPサーバーに接続中...")
            session = ClientSession(transport[0], transport[1])
            print("MCPサーバーに接続しました。初期化を開始します...")
            
            # 単純なawaitを使用
            try:
                print("initialize()を呼び出します...")
                init_result = await session.initialize()
                print(f"MCPサーバーの初期化完了。バージョン: {init_result.protocolVersion}")
                
                # ツール一覧を取得
                print("ツール一覧の取得を開始します...")
                tools_response = await session.list_tools()
                tools = tools_response.tools if hasattr(tools_response, "tools") else []
                
                # ツール一覧を表示
                print(f"GitHub MCPサーバーの利用可能な関数（ツール）: {len(tools)}個")
                for tool in tools:
                    print(f"- {tool.name}: {tool.description}")
                
                return tools
            except RuntimeError as re:
                print(f"MCPサーバー初期化時のRuntimeError: {re}")
                # バージョン問題であれば、ここで対処
                if "Unsupported protocol version" in str(re):
                    print("プロトコルバージョンの不一致が検出されました。")
                    # 必要に応じてバージョンチェックをパッチする処理を追加
                return []
    except Exception as e:
        print(f"MCPサーバー接続中にエラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()
        return []
    finally:
        print("list_mcp_tools関数の処理を完了しました")

async def main():
    print("Azure OpenAIとGitHub MCPサーバーの接続を開始します。")
    # ツール一覧を取得
    tools = await list_mcp_tools()
    print("ツール一覧取得完了。")
    # 必要に応じてAzure OpenAIでツールを活用した処理を追加
    # 例: ツール一覧を基にプロンプトを生成して送信
    if tools:
        prompt = "以下のツールが利用可能です。どのように使いますか？\n" + "\n".join([t.name for t in tools])
        response = await client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "あなたは役立つアシスタントです。"},
                {"role": "user", "content": prompt}
            ]
        )
        print(f"Azure OpenAIからの応答: {response.choices[0].message.content}")

# イベントループを実行
if __name__ == "__main__":
    asyncio.run(main())