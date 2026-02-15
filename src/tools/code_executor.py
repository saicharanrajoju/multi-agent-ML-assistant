import os
from e2b_code_interpreter import Sandbox
from src.config import E2B_API_KEY

class CodeExecutor:
    """Executes Python code safely in an E2B cloud sandbox."""

    def __init__(self):
        """Initialize the E2B sandbox with data science packages pre-installed."""
        self.sandbox = None

    def start(self):
        """Start the sandbox and install required packages."""
        print("📦 Starting E2B sandbox...")
        self.sandbox = Sandbox.create(api_key=E2B_API_KEY)

        # Install data science packages in the sandbox
        print("📚 Installing packages in sandbox...")
        self.sandbox.commands.run(
            "pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm shap joblib",
            timeout=120
        )
        print("✅ Sandbox ready!")

    def execute_code(self, code: str, timeout: int = 120) -> dict:
        """
        Execute Python code in the E2B sandbox.

        Args:
            code: Python code string to execute
            timeout: Max execution time in seconds

        Returns:
            dict with keys:
                - success (bool): whether code ran without errors
                - stdout (str): standard output
                - stderr (str): standard error
                - results (list): rich outputs (plots, dataframes, etc.)
                - error (str | None): error message if failed
        """
        if not self.sandbox:
            self.start()

        try:
            execution = self.sandbox.run_code(code, timeout=timeout)

            stdout = ""
            stderr = ""
            results = []

            # Collect text output
            for log in execution.logs.stdout:
                stdout += log + "\n"
            for log in execution.logs.stderr:
                stderr += log + "\n"

            # Collect rich results (charts, dataframes, etc.)
            for result in execution.results:
                if hasattr(result, 'text') and result.text:
                    results.append({"type": "text", "data": result.text})
                if hasattr(result, 'png') and result.png:
                    results.append({"type": "png", "data": result.png})

            if execution.error:
                return {
                    "success": False,
                    "stdout": stdout.strip(),
                    "stderr": stderr.strip(),
                    "results": results,
                    "error": f"{execution.error.name}: {execution.error.value}\n{execution.error.traceback}",
                }

            return {
                "success": True,
                "stdout": stdout.strip(),
                "stderr": stderr.strip(),
                "results": results,
                "error": None,
            }

        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": "",
                "results": [],
                "error": f"Sandbox execution error: {str(e)}",
            }

    def upload_file(self, local_path: str, sandbox_path: str = None) -> str:
        """
        Upload a file to the sandbox.

        Args:
            local_path: path to the local file
            sandbox_path: destination path in sandbox (default: /home/user/{filename})

        Returns:
            The path of the file inside the sandbox
        """
        if not self.sandbox:
            self.start()

        filename = os.path.basename(local_path)
        if sandbox_path is None:
            sandbox_path = f"/home/user/{filename}"

        with open(local_path, "rb") as f:
            self.sandbox.files.write(sandbox_path, f)

        print(f"📁 Uploaded {local_path} → {sandbox_path}")
        return sandbox_path

    def download_file(self, sandbox_path: str, local_path: str):
        """Download a file from the sandbox to local filesystem."""
        if not self.sandbox:
            raise RuntimeError("Sandbox not started")

        content = self.sandbox.files.read(sandbox_path, format="bytes")
        with open(local_path, "wb") as f:
            f.write(content)
        print(f"📥 Downloaded {sandbox_path} → {local_path}")

    def close(self):
        """Shut down the sandbox."""
        if self.sandbox:
            self.sandbox.kill()
            self.sandbox = None
            print("🛑 Sandbox closed")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class SandboxManager:
    """Manages a single shared E2B sandbox across all agents."""

    _instance = None
    _sandbox = None
    _initialized = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_sandbox(self):
        """Get or create the shared sandbox."""
        if self._sandbox is None:
            print("📦 Starting shared E2B sandbox...")
            self._sandbox = Sandbox.create(api_key=E2B_API_KEY)
            if not self._initialized:
                print("📚 Installing packages in shared sandbox...")
                self._sandbox.commands.run(
                    "pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm shap joblib",
                    timeout=120
                )
                self._initialized = True
                print("✅ Shared sandbox ready!")
        return self._sandbox

    def execute_code(self, code: str, timeout: int = 300) -> dict:
        """Execute code in the shared sandbox."""
        sandbox = self.get_sandbox()
        try:
            execution = sandbox.run_code(code, timeout=timeout)

            stdout = "\n".join(execution.logs.stdout)
            stderr = "\n".join(execution.logs.stderr)
            results = []

            for result in execution.results:
                if hasattr(result, 'text') and result.text:
                    results.append({"type": "text", "data": result.text})
                if hasattr(result, 'png') and result.png:
                    results.append({"type": "png", "data": result.png})

            if execution.error:
                return {
                    "success": False,
                    "stdout": stdout.strip(),
                    "stderr": stderr.strip(),
                    "results": results,
                    "error": f"{execution.error.name}: {execution.error.value}\n{execution.error.traceback}",
                }

            return {
                "success": True,
                "stdout": stdout.strip(),
                "stderr": stderr.strip(),
                "results": results,
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": "",
                "results": [],
                "error": f"Sandbox execution error: {str(e)}",
            }

    def upload_file(self, local_path: str, sandbox_path: str = None) -> str:
        """Upload file to shared sandbox."""
        sandbox = self.get_sandbox()
        filename = os.path.basename(local_path)
        if sandbox_path is None:
            sandbox_path = f"/home/user/{filename}"
        with open(local_path, "rb") as f:
            sandbox.files.write(sandbox_path, f)
        print(f"  📁 Uploaded {filename} → {sandbox_path}")
        return sandbox_path

    def download_file(self, sandbox_path: str, local_path: str):
        """Download file from shared sandbox."""
        sandbox = self.get_sandbox()
        content = sandbox.files.read(sandbox_path, format="bytes")
        with open(local_path, "wb") as f:
            f.write(content)
        print(f"  📥 Downloaded → {local_path}")

    def close(self):
        """Shut down the shared sandbox."""
        if self._sandbox:
            self._sandbox.kill()
            self._sandbox = None
            self._initialized = False
            print("🛑 Shared sandbox closed")

    @classmethod
    def reset(cls):
        """Reset the singleton (useful for testing)."""
        if cls._instance:
            cls._instance.close()
        cls._instance = None


def get_shared_sandbox() -> SandboxManager:
    return SandboxManager.get_instance()
