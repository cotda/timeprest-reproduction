"""n5k/kaggle/kernel_lib.run_step: a step whose worker outlives the launcher (torchrun starts each worker in a
session of its own, out of reach of `timeout`) must not hang the kernel by holding the output pipe open."""
import os
import shutil
import subprocess
import sys
import textwrap
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "n5k", "kaggle"))
import kernel_lib  # noqa: E402


def coreutils_timeout():
    if os.name != "nt":
        return shutil.which("timeout")
    for d in (r"C:\Program Files\Git\usr\bin", r"C:\Program Files (x86)\Git\usr\bin"):
        if os.path.exists(os.path.join(d, "timeout.exe")):
            return os.path.join(d, "timeout.exe")
    return None


@pytest.fixture(autouse=True)
def gnu_timeout(monkeypatch):
    t = coreutils_timeout()
    if t is None:
        pytest.skip("coreutils timeout not found")
    monkeypatch.setattr(kernel_lib, "TIMEOUT", t)


def test_step_output_and_exit_code(tmp_path):
    code, lines = kernel_lib.run_step([sys.executable, "-c", "print('a'); print('X_JSON 1'); raise SystemExit(3)"],
                                      30, str(tmp_path / "s.log"), poll_s=0.2)
    assert code == 3 and [l.strip() for l in lines] == ["a", "X_JSON 1"]
    assert open(tmp_path / "s.log").read().split() == ["a", "X_JSON", "1"]


def test_orphaned_worker_does_not_hang(tmp_path):
    flag = "start_new_session=True" if os.name != "nt" else "creationflags=subprocess.CREATE_NEW_PROCESS_GROUP"
    script = tmp_path / "launcher.py"
    script.write_text(textwrap.dedent(f"""
        import subprocess, sys, time
        subprocess.Popen([sys.executable, "-c", "import time; print('worker up', flush=True); time.sleep(300)"],
                         {flag})
        print("launcher up", flush=True)
        time.sleep(300)
    """))
    t0 = time.time()
    code, lines = kernel_lib.run_step([sys.executable, str(script)], 3, str(tmp_path / "s.log"), grace_s=2, poll_s=0.2)
    assert time.time() - t0 < 30
    assert code != 0
    assert any("launcher up" in l for l in lines)
