import os
import threading
from e2b_code_interpreter import Sandbox
from src.config import E2B_API_KEY

SANDBOX_PACKAGES = "pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm shap joblib imbalanced-learn"

# Files to checkpoint locally after each stage completes.
# On sandbox crash, these are re-uploaded to the fresh sandbox automatically.
STAGE_CHECKPOINTS = {
    "profiler":          [],
    "cleaner":           ["/home/user/cleaned_data.csv"],
    "feature_engineer":  ["/home/user/featured_data.csv"],
    "modeler":           ["/home/user/best_model.joblib", "/home/user/preprocessor.joblib"],
    "deployer":          [],  # deployer downloads to local outputs/ itself
}


class SandboxManager:
    """One isolated E2B sandbox per pipeline run, with crash recovery via stage checkpoints."""

    def __init__(self, run_id: str, checkpoint_dir: str = None):
        self.run_id = run_id
        self.checkpoint_dir = checkpoint_dir or os.path.join("outputs", "checkpoints", run_id)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._sandbox = None
        self._lock = threading.Lock()

    def _create_sandbox(self):
        print(f"📦 [Sandbox:{self.run_id}] Starting E2B sandbox...")
        try:
            self._sandbox = Sandbox.create(api_key=E2B_API_KEY, timeout=3600)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create E2B sandbox: {str(e)[:200]}. "
                "Check your E2B_API_KEY and network connection."
            )
        print(f"📚 [Sandbox:{self.run_id}] Installing packages...")
        try:
            self._sandbox.commands.run(f"pip install {SANDBOX_PACKAGES}", timeout=180)
        except Exception as e:
            print(f"⚠️ [Sandbox:{self.run_id}] Package install warning: {str(e)[:200]}")
        print(f"✅ [Sandbox:{self.run_id}] Sandbox ready!")

    def _is_alive(self) -> bool:
        if self._sandbox is None:
            return False
        try:
            self._sandbox.files.list("/home/user")
            return True
        except Exception:
            return False

    def get_sandbox(self):
        with self._lock:
            if not self._is_alive():
                self._create_sandbox()
                self._restore_checkpoints()
            return self._sandbox

    def _restore_checkpoints(self):
        """Re-upload locally saved files after a sandbox crash or restart."""
        for fname in os.listdir(self.checkpoint_dir):
            local = os.path.join(self.checkpoint_dir, fname)
            sandbox_path = f"/home/user/{fname}"
            try:
                with open(local, "rb") as f:
                    self._sandbox.files.write(sandbox_path, f)
                print(f"  🔄 [Sandbox:{self.run_id}] Restored checkpoint: {fname}")
            except Exception as e:
                print(f"  ⚠️ [Sandbox:{self.run_id}] Could not restore {fname}: {e}")

    def checkpoint(self, stage: str):
        """Download stage output files locally so they survive a sandbox crash."""
        for sandbox_path in STAGE_CHECKPOINTS.get(stage, []):
            fname = os.path.basename(sandbox_path)
            local = os.path.join(self.checkpoint_dir, fname)
            try:
                content = self._sandbox.files.read(sandbox_path, format="bytes")
                with open(local, "wb") as f:
                    f.write(content)
                print(f"  💾 [Sandbox:{self.run_id}] Checkpointed: {fname}")
            except Exception as e:
                print(f"  ⚠️ [Sandbox:{self.run_id}] Checkpoint failed for {fname}: {e}")

    def execute_code(self, code: str, timeout: int = 300) -> dict:
        sandbox = self.get_sandbox()
        try:
            execution = sandbox.run_code(code, timeout=timeout)

            stdout = "\n".join(execution.logs.stdout)
            stderr = "\n".join(execution.logs.stderr)
            results = []

            for result in execution.results:
                if hasattr(result, "text") and result.text:
                    results.append({"type": "text", "data": result.text})
                if hasattr(result, "png") and result.png:
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
            error_str = str(e)
            if "502" in error_str or "sandbox was not found" in error_str.lower():
                with self._lock:
                    self._sandbox = None  # Mark dead; next call recreates + restores
            return {
                "success": False,
                "stdout": "",
                "stderr": "",
                "results": [],
                "error": f"Sandbox execution error: {error_str[:500]}",
            }

    def upload_file(self, local_path: str, sandbox_path: str = None) -> str:
        sandbox = self.get_sandbox()
        filename = os.path.basename(local_path)
        if sandbox_path is None:
            sandbox_path = f"/home/user/{filename}"
        with open(local_path, "rb") as f:
            sandbox.files.write(sandbox_path, f)
        print(f"  📁 [Sandbox:{self.run_id}] Uploaded {filename} -> {sandbox_path}")
        return sandbox_path

    def download_file(self, sandbox_path: str, local_path: str):
        sandbox = self.get_sandbox()
        content = sandbox.files.read(sandbox_path, format="bytes")
        with open(local_path, "wb") as f:
            f.write(content)
        print(f"  📥 [Sandbox:{self.run_id}] Downloaded -> {local_path}")

    def close(self):
        with self._lock:
            if self._sandbox:
                try:
                    self._sandbox.kill()
                except Exception:
                    pass
                self._sandbox = None
                print(f"🛑 [Sandbox:{self.run_id}] Sandbox closed")

    @classmethod
    def reset(cls):
        with _registry_lock:
            for sm in _registry.values():
                sm.close()
            _registry.clear()



# --- Per-run registry ---

_registry: dict[str, SandboxManager] = {}
_registry_lock = threading.Lock()


def get_sandbox_for_run(run_id: str, checkpoint_dir: str = None) -> SandboxManager:
    """Get or create the SandboxManager for a specific pipeline run (thread-safe)."""
    with _registry_lock:
        if run_id not in _registry:
            _registry[run_id] = SandboxManager(run_id, checkpoint_dir)
        return _registry[run_id]


def close_sandbox_for_run(run_id: str):
    """Shut down and deregister the sandbox when a pipeline run completes or fails."""
    with _registry_lock:
        if run_id in _registry:
            _registry[run_id].close()
            del _registry[run_id]


# Backwards-compatible shim — existing agents keep working without changes.
# Migrate agents one by one to get_sandbox_for_run(run_id) using LangGraph's thread_id.
def get_shared_sandbox() -> SandboxManager:
    return get_sandbox_for_run("default")
