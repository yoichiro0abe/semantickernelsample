import sys
import os
import subprocess
import threading
import time
from datetime import datetime

# ログファイルパス
log_dir = os.path.dirname(os.path.abspath(__file__))
stdin_log_path = os.path.join(log_dir, "stdin_log.txt")
stdout_log_path = os.path.join(log_dir, "stdout_log.txt")

# 入出力の記録ファイル
in_log = open(stdin_log_path, "w", encoding="utf-8")
out_log = open(stdout_log_path, "w", encoding="utf-8")

print(f"ログファイル: {stdin_log_path}, {stdout_log_path}", file=sys.stderr)

# 起動時刻をログに記録
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
in_log.write(f"[{timestamp}] プロキシ起動\n")
in_log.flush()
out_log.write(f"[{timestamp}] プロキシ起動\n")
out_log.flush()

# マルチスレッド実行のためのロック
in_lock = threading.Lock()
out_lock = threading.Lock()

def log_message(log_file, lock, prefix, message):
    """スレッドセーフにログを記録"""
    with lock:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_file.write(f"[{timestamp}] {prefix}: {message}")
        log_file.flush()

# 標準入力からMCPサーバーへデータを転送
def stdin_to_server(process):
    try:
        log_message(in_log, in_lock, "INFO", "標準入力転送開始\n")
        
        # バイナリモードでないと正しく動作しない可能性がある
        for line in sys.stdin.buffer:
            try:
                # データをログ
                decoded = line.decode("utf-8", errors="replace")
                log_message(in_log, in_lock, "TO_SERVER", decoded)
                
                # データを転送
                process.stdin.write(line)
                process.stdin.flush()
            except Exception as e:
                log_message(in_log, in_lock, "ERROR", f"標準入力転送中エラー: {e}\n")
                break
    except Exception as e:
        log_message(in_log, in_lock, "ERROR", f"標準入力転送スレッドエラー: {e}\n")
    finally:
        log_message(in_log, in_lock, "INFO", "標準入力転送終了\n")

# MCPサーバーから標準出力へデータを転送
def server_to_stdout(process):
    try:
        log_message(out_log, out_lock, "INFO", "標準出力転送開始\n")
        
        for line in process.stdout:
            try:
                # データをログ
                decoded = line.decode("utf-8", errors="replace")
                log_message(out_log, out_lock, "FROM_SERVER", decoded)
                
                # データを転送
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
            except Exception as e:
                log_message(out_log, out_lock, "ERROR", f"標準出力転送中エラー: {e}\n")
                break
    except Exception as e:
        log_message(out_log, out_lock, "ERROR", f"標準出力転送スレッドエラー: {e}\n")
    finally:
        log_message(out_log, out_lock, "INFO", "標準出力転送終了\n")

# エラー出力を転送
def server_to_stderr(process):
    try:
        log_message(out_log, out_lock, "INFO", "エラー出力転送開始\n")
        
        for line in process.stderr:
            try:
                # データをログ
                decoded = line.decode("utf-8", errors="replace")
                log_message(out_log, out_lock, "SERVER_ERR", decoded)
                
                # データを転送
                sys.stderr.buffer.write(line)
                sys.stderr.buffer.flush()
            except Exception as e:
                log_message(out_log, out_lock, "ERROR", f"エラー出力転送中エラー: {e}\n")
                break
    except Exception as e:
        log_message(out_log, out_lock, "ERROR", f"エラー出力転送スレッドエラー: {e}\n")
    finally:
        log_message(out_log, out_lock, "INFO", "エラー出力転送終了\n")

# メイン処理
def main():
    try:
        print("MCPプロキシを起動しています...", file=sys.stderr)
        
        # 環境変数をログに記録
        in_log.write("環境変数:\n")
        for key, value in os.environ.items():
            if key in ["PATH", "GITHUB_TOKEN"]:
                value_to_show = "***" if key == "GITHUB_TOKEN" else value[:20] + "..."
                in_log.write(f"  {key}={value_to_show}\n")
        in_log.flush()
        
        # 実際のMCPサーバーを起動
        cmd = "npx --yes @modelcontextprotocol/server-github"
        in_log.write(f"実行コマンド: {cmd}\n")
        in_log.flush()
        
        # サブプロセスを起動
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            env={**os.environ, "GITHUB_TOKEN": os.getenv("GITHUB_TOKEN", "")}
        )
        
        in_log.write(f"プロセスID: {process.pid}\n")
        in_log.flush()
        
        # 転送スレッドを起動
        threads = [
            threading.Thread(target=stdin_to_server, args=(process,)),
            threading.Thread(target=server_to_stdout, args=(process,)),
            threading.Thread(target=server_to_stderr, args=(process,))
        ]
        
        # スレッドを開始
        for thread in threads:
            thread.daemon = True
            thread.start()
        
        # プロセスの終了を待機
        process.wait()
        
    except Exception as e:
        print(f"プロキシエラー: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
    finally:
        print("MCPプロキシ処理終了", file=sys.stderr)
        in_log.close()
        out_log.close()

if __name__ == "__main__":
    main()