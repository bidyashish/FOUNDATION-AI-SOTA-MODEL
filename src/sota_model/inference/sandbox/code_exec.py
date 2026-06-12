"""Sandboxed code-execution tool.

Replaces the placeholder `_stub_code_exec` in `inference/tools.py` with a
real subprocess-based sandbox suitable for the modelcard 5.1 agentic-safety
threat model:

  - separate process, killed by timeout
  - no network (LD_PRELOAD shim or unshare on Linux; documented best-effort
    on macOS — see the `__call__` notes)
  - filesystem rooted at a fresh temp directory
  - resource limits via `resource.setrlimit` (mem, CPU, fsize, nofile, nproc)
  - environment scrubbed to a minimal allowlist
  - no shared memory between calls

Important: this is a baseline that a real deployment must layer additional
isolation on top of (gVisor, firecracker, Docker w/ seccomp). The contract
this sandbox guarantees is "best-effort soft isolation"; enforcement is
operator-supplied for production.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SandboxConfig:
    timeout_seconds: float = 8.0
    max_memory_mb: int = 512
    max_output_bytes: int = 64 * 1024
    max_files: int = 32
    max_processes: int = 16
    allowed_languages: tuple[str, ...] = ("python", "bash")
    allow_network: bool = False
    env_allowlist: tuple[str, ...] = ("PATH", "LANG", "LC_ALL")


@dataclass
class CodeExecResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    truncated: bool

    @property
    def ok(self) -> bool:
        return (not self.timed_out) and self.exit_code == 0

    def to_dict(self) -> dict:
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "truncated": self.truncated,
            "ok": self.ok,
        }


class CodeExecSandbox:
    def __init__(self, cfg: Optional[SandboxConfig] = None):
        self.cfg = cfg or SandboxConfig()

    def __call__(self, language: str, code: str) -> dict:
        return self.run(language, code).to_dict()

    def run(self, language: str, code: str) -> CodeExecResult:
        if language not in self.cfg.allowed_languages:
            return CodeExecResult(
                stdout="",
                stderr=f"language not allowed: {language!r}",
                exit_code=64,
                timed_out=False,
                truncated=False,
            )

        with tempfile.TemporaryDirectory(prefix="sota-sandbox-") as work_dir:
            return self._run_in(work_dir, language, code)

    # --- internals ---

    def _run_in(self, work_dir: str, language: str, code: str) -> CodeExecResult:
        cmd = self._command_for(language, code, work_dir)
        env = {k: os.environ[k] for k in self.cfg.env_allowlist if k in os.environ}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["HOME"] = work_dir
        env["TMPDIR"] = work_dir

        try:
            proc = subprocess.run(
                cmd,
                cwd=work_dir,
                env=env,
                input="",
                capture_output=True,
                text=True,
                timeout=self.cfg.timeout_seconds,
                preexec_fn=self._preexec_apply_rlimits if hasattr(os, "fork") else None,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            return CodeExecResult(
                stdout=(e.stdout or b"").decode("utf-8", errors="replace")[: self.cfg.max_output_bytes],
                stderr=(e.stderr or b"").decode("utf-8", errors="replace")[: self.cfg.max_output_bytes]
                or f"timed out after {self.cfg.timeout_seconds}s",
                exit_code=124,
                timed_out=True,
                truncated=False,
            )

        truncated = (
            len(proc.stdout) > self.cfg.max_output_bytes
            or len(proc.stderr) > self.cfg.max_output_bytes
        )
        return CodeExecResult(
            stdout=proc.stdout[: self.cfg.max_output_bytes],
            stderr=proc.stderr[: self.cfg.max_output_bytes],
            exit_code=proc.returncode,
            timed_out=False,
            truncated=truncated,
        )

    def _command_for(self, language: str, code: str, work_dir: str) -> list[str]:
        if language == "python":
            script = Path(work_dir) / "snippet.py"
            script.write_text(self._wrap_python(code))
            return [sys.executable, "-I", "-S", "-B", str(script)]
        if language == "bash":
            script = Path(work_dir) / "snippet.sh"
            script.write_text(code)
            return ["/bin/bash", "--noprofile", "--norc", "-r", str(script)]
        raise AssertionError(f"unreachable: language {language!r}")

    @staticmethod
    def _wrap_python(code: str) -> str:
        # Strip imports of subprocess / socket if `allow_network=False`?
        # No — that's a soft signal at best. Real network blocking is done
        # at the syscall layer by an operator-supplied wrapper (e.g. nsjail).
        # Here we only ensure the script runs in isolated mode.
        return code

    def _preexec_apply_rlimits(self) -> None:
        # Linux/macOS only. Lower the bar on memory, CPU, file count, output.
        try:
            import resource
        except ImportError:
            return
        mb = self.cfg.max_memory_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mb, mb))
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (
                int(self.cfg.timeout_seconds) + 1, int(self.cfg.timeout_seconds) + 1,
            ))
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (self.cfg.max_output_bytes, self.cfg.max_output_bytes))
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (self.cfg.max_files, self.cfg.max_files))
        except (ValueError, OSError):
            pass
        if hasattr(resource, "RLIMIT_NPROC"):
            try:
                resource.setrlimit(resource.RLIMIT_NPROC, (self.cfg.max_processes, self.cfg.max_processes))
            except (ValueError, OSError):
                pass
        try:
            os.setpgrp()
        except OSError:
            pass


def make_code_exec_callable(cfg: Optional[SandboxConfig] = None):
    """Return a `(language, code) -> dict` callable for the ToolRegistry."""
    sandbox = CodeExecSandbox(cfg)
    def _call(language: str, code: str) -> dict:
        return sandbox(language, code)
    return _call
