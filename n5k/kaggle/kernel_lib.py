"""Shared runner of the n5k Kaggle kernels (n5k/kaggle/runK/runK.py). Ships inside the code dataset
kaggleolap/timeprest-code; each kernel script only locates the code and calls `run(...)`.

Kaggle returns no log while a committed kernel runs (and none at all for a cancelled one), so every
step has a time limit (coreutils `timeout`) and the whole kernel ends before `budget_min`: the log,
RUN_SUMMARY.json and the outputs are then always saved.

Outputs (/kaggle/working): logs/<step>.log, n5k_results/ (no checkpoints), n5k_cache/ (when built),
RUN_SUMMARY.json (status, minutes and the last *_JSON line of every step).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile

WORK = "/kaggle/working"
CODE = os.path.join(WORK, "code")
LOGS = os.path.join(WORK, "logs")
TIMEOUT = "timeout"                       # coreutils timeout (the tests point it to Git's on Windows)
T0 = time.time()


def say(*a):
    print(time.strftime("%H:%M:%S"), f"[+{(time.time() - T0) / 60:.1f}m]", *a, flush=True)


def find(name, top="/kaggle/input", max_depth=6, want_dir=False):
    """First file (or directory) `name` under `top`, without walking the Nutrition5k image folders."""
    base = top.rstrip("/").count("/")
    for root, dirs, files in os.walk(top, followlinks=True):
        if name in (dirs if want_dir else files):
            return os.path.join(root, name)
        dirs[:] = sorted(d for d in dirs if d != "imagery") if root.count("/") - base < max_depth else []
    return None


def setup_code(src: str):
    """Copy the code (extracted folder or zip) to CODE and install it in editable mode."""
    if src.endswith(".zip"):
        with zipfile.ZipFile(src) as f:
            f.extractall(CODE)
    else:
        shutil.copytree(src, CODE, dirs_exist_ok=True)
    os.chdir(CODE)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", "."], check=True)


def restore_results():
    """n5k_results/ of an attached earlier kernel (checks file, sweep metrics) -> CODE/n5k/results."""
    prev = find("n5k_results", want_dir=True)
    if prev:
        shutil.copytree(prev, os.path.join(CODE, "n5k", "results"), dirs_exist_ok=True)
    return prev


def _descendants(pid: int) -> list:
    try:
        import psutil
        return psutil.Process(pid).children(recursive=True)
    except Exception:                         # no psutil, or the process is gone
        return []


def _kill(procs):
    for q in procs:
        try:
            q.kill()
        except Exception:
            pass


def run_step(cmd, limit_s: float, log_path: str, env=None, grace_s: float = 60.0, poll_s: float = 5.0):
    """Run `cmd` under coreutils `timeout` (TERM at limit_s, KILL grace_s later), teeing its output to
    log_path and stdout. -> (exit code, output lines).

    torchrun starts each worker in a session of its own, which `timeout` (it signals its own process
    group) does not reach: a worker that outlives torchrun keeps the output pipe open and reading
    until EOF would never end. So every descendant seen while the step runs is recorded and, once
    `timeout` has exited (or at the hard deadline), those still alive are killed."""
    lines, seen = [], {}
    p = subprocess.Popen([TIMEOUT, "-k", f"{int(grace_s)}", f"{int(limit_s)}s", *cmd], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, env=env)

    def reader():
        with open(log_path, "w") as log:
            for line in p.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
                lines.append(line)

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    deadline = time.time() + limit_s + 2 * grace_s
    while p.poll() is None and time.time() < deadline:
        for q in _descendants(p.pid):
            seen[q.pid] = q
        time.sleep(poll_s)
    if p.poll() is None:                      # `timeout` itself did not end the step
        seen.update({q.pid: q for q in _descendants(p.pid)})
        p.kill()
    code = p.wait()
    _kill(seen.values())
    t.join(grace_s)
    if t.is_alive():                          # something unseen still holds the pipe: leave the (daemon) reader
        lines.append("[kernel_lib] output pipe still open after the step ended; stopped reading\n")
    return code, lines


def save_results():
    """CODE/n5k/results (no checkpoints) -> WORK/n5k_results; after every step, so a kernel stopped by
    Kaggle still has the results of the steps that finished."""
    if os.path.isdir(os.path.join(CODE, "n5k", "results")):
        shutil.copytree(os.path.join(CODE, "n5k", "results"), os.path.join(WORK, "n5k_results"), dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("*.pt", "*.pt.tmp", "work"))


def run(steps, stop_if_failed=(), budget_min=600.0, restore=False, margin_min=8.0):
    """steps = [(name, cmd, limit_min)]. A step gets min(limit, budget left - margin) minutes; exit
    code 124/137 of `timeout` is recorded as status "timeout"."""
    summary = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "budget_min": budget_min, "steps": []}

    def save():
        with open(os.path.join(WORK, "RUN_SUMMARY.json"), "w") as f:
            json.dump(summary, f, indent=1)

    os.makedirs(LOGS, exist_ok=True)
    commit = find("COMMIT.txt")
    summary["commit"] = open(commit).read().strip() if commit else None
    src = os.path.dirname(find("pyproject.toml") or "") or find("timeprest_code.zip")
    say("code", summary["commit"], "from", src)
    setup_code(src)
    if restore:
        say("restored results from", restore_results())
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if shutil.which("nvidia-smi"):
        say("gpus:", subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip())
    save()

    for name, cmd, limit_min in steps:
        left = budget_min - margin_min - (time.time() - T0) / 60
        lim = min(limit_min, left)
        if left < 1:
            summary["steps"].append({"step": name, "status": "skipped (no time left)"})
            say(f"skip {name}: no time left")
            continue
        say(f"===== {name} (limit {lim:.0f} min): {' '.join(cmd)}")
        t0 = time.time()
        code, lines = run_step(cmd, lim * 60, os.path.join(LOGS, f"{name}.log"), env)
        status = "ok" if code == 0 else "timeout" if code in (124, 137, -9) else "failed"
        rec = {"step": name, "status": status, "exit": code, "minutes": round((time.time() - t0) / 60, 1),
               "json": [l.strip() for l in lines if "_JSON " in l][-1:], "tail": [l.rstrip() for l in lines[-15:]]}
        summary["steps"].append(rec)
        save_results()
        save()
        say(f"===== {name}: {status} (exit {code}, {rec['minutes']} min)")
        if status != "ok" and name in stop_if_failed:
            say(f"step {name} {status}: stopping")
            break

    save_results()
    os.chdir(WORK)
    shutil.rmtree(CODE, ignore_errors=True)        # keep the kernel output small (code = the input dataset)
    summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save()
    say("KERNEL_DONE", json.dumps([{k: s.get(k) for k in ("step", "status", "minutes")} for s in summary["steps"]]))
