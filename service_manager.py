"""MuseTalk 服务管理器

用于启动、停止和管理 MuseTalk API 服务。
"""

import os
import sys
import time
import signal
import subprocess
from pathlib import Path
from typing import Optional

import requests


class MuseTalkServiceManager:
    """MuseTalk 服务管理器"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8770
    ):
        """初始化服务管理器

        Args:
            host: 服务主机地址
            port: 服务端口
        """
        self.host = host
        self.port = port
        self.service_url = f"http://{host}:{port}"

        # 服务目录
        self.service_dir = Path(__file__).parent
        self.venv_python = self.service_dir / "venv" / "bin" / "python"
        self.api_script = self.service_dir / "api_server.py"

        # 进程句柄
        self.process: Optional[subprocess.Popen] = None

        # 禁用代理
        self.proxies = {
            'http': '',
            'https': '',
            'no_proxy': 'localhost,127.0.0.1'
        }

    def is_running(self) -> bool:
        """检查服务是否运行中

        Returns:
            bool: 服务是否运行
        """
        try:
            response = requests.get(
                f"{self.service_url}/health",
                timeout=5,
                proxies=self.proxies
            )
            return response.status_code == 200
        except:
            return False

    def start(self, wait: bool = True, timeout: int = 120) -> bool:
        """启动服务

        Args:
            wait: 是否等待服务就绪
            timeout: 等待超时时间（秒），MuseTalk 模型加载较慢

        Returns:
            bool: 启动是否成功
        """
        if self.is_running():
            print("MuseTalk 服务已在运行中")
            return True

        if not self.venv_python.exists():
            print(f"错误: Python 解释器未找到: {self.venv_python}")
            return False

        if not self.api_script.exists():
            print(f"错误: API 脚本未找到: {self.api_script}")
            return False

        print(f"正在启动 MuseTalk 服务 ({self.host}:{self.port})...")

        # 设置环境变量
        env = os.environ.copy()
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        # 日志文件
        self.log_file = self.service_dir / "musetalk_server.log"
        log_handle = open(self.log_file, 'a')

        print(f"日志文件: {self.log_file}")

        # 启动进程
        self.process = subprocess.Popen(
            [
                str(self.venv_python),
                str(self.api_script),
                "-a", self.host,
                "-p", str(self.port)
            ],
            cwd=str(self.service_dir),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True
        )

        if wait:
            # 等待服务就绪
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self.is_running():
                    print(f"✓ MuseTalk 服务已启动 (PID: {self.process.pid})")
                    return True
                time.sleep(2)

            print(f"✗ MuseTalk 服务启动超时")
            self.stop()
            return False

        return True

    def stop(self) -> bool:
        """停止服务

        Returns:
            bool: 停止是否成功
        """
        if not self.is_running():
            print("MuseTalk 服务未运行")
            return True

        print("正在停止 MuseTalk 服务...")

        try:
            # 尝试通过 API 优雅关闭
            response = requests.post(
                f"{self.service_url}/shutdown",
                timeout=5,
                proxies=self.proxies
            )
            time.sleep(2)

            if not self.is_running():
                print("✓ MuseTalk 服务已停止")
                return True

        except:
            pass

        # 如果 API 关闭失败，尝试杀进程
        if self.process is not None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                time.sleep(2)

                if self.process.poll() is None:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)

                print("✓ MuseTalk 服务已停止 (强制)")
                return True
            except:
                pass

        # 尝试通过端口查找并杀进程
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{self.port}"],
                capture_output=True,
                text=True
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                    except:
                        pass
                time.sleep(1)
                print("✓ MuseTalk 服务已停止")
                return True
        except:
            pass

        return not self.is_running()

    def restart(self) -> bool:
        """重启服务

        Returns:
            bool: 重启是否成功
        """
        self.stop()
        time.sleep(2)
        return self.start()

    def get_status(self) -> dict:
        """获取服务状态

        Returns:
            dict: 服务状态信息
        """
        running = self.is_running()
        status = {
            "running": running,
            "host": self.host,
            "port": self.port,
            "url": self.service_url
        }

        if running:
            try:
                response = requests.get(
                    f"{self.service_url}/health",
                    timeout=5,
                    proxies=self.proxies
                )
                if response.status_code == 200:
                    status.update(response.json())
            except:
                pass

        return status


# 命令行入口
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MuseTalk 服务管理")
    parser.add_argument("action", choices=["start", "stop", "restart", "status"],
                        help="操作: start/stop/restart/status")
    parser.add_argument("--host", default="127.0.0.1", help="服务器地址")
    parser.add_argument("--port", type=int, default=8770, help="服务器端口")

    args = parser.parse_args()

    manager = MuseTalkServiceManager(host=args.host, port=args.port)

    if args.action == "start":
        success = manager.start()
        sys.exit(0 if success else 1)

    elif args.action == "stop":
        success = manager.stop()
        sys.exit(0 if success else 1)

    elif args.action == "restart":
        success = manager.restart()
        sys.exit(0 if success else 1)

    elif args.action == "status":
        status = manager.get_status()
        print(f"服务状态: {'运行中' if status['running'] else '未运行'}")
        print(f"地址: {status['url']}")
        if status.get('model_loaded'):
            print(f"模型: 已加载")
        sys.exit(0)
