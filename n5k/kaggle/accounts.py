"""Several Kaggle accounts for the n5k kernels (Kaggle runs at most 2 GPU sessions per account).

Tokens live in kaggle_accounts.json at the repo root (git-ignored, never committed or printed):
    {"default": "kaggleolap", "accounts": {"kaggleolap": {"token": "KGAT_..."}, "<user>": {"token": "KGAT_..."}}}
The account name is the Kaggle username (it is the owner part of every dataset / kernel id).

    python n5k/kaggle/accounts.py list
    python n5k/kaggle/accounts.py add <user> --token-file token.txt     # first KGAT_... token in the file
    python n5k/kaggle/accounts.py whoami <user>                         # check the token works
    python n5k/kaggle/accounts.py run <user> -- kaggle kernels status <user>/n5k-run12-tp-lr
    python n5k/kaggle/accounts.py push-code <user> -m "msg"             # -> <user>/timeprest-code
    python n5k/kaggle/accounts.py push-kernel <user> n5k/kaggle/run12_tp_lr [--cpu]

`run` sets KAGGLE_API_TOKEN (and N5K_KAGGLE_USER, read by push_code.py) for that one command only.
`push-kernel` pushes a copy of the kernel folder whose kernel-metadata.json has every "kaggleolap/" id
(kernel id, code dataset, attached kernels) moved to the chosen account; the folder itself is unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STORE = os.path.join(REPO, "kaggle_accounts.json")
HOME_OWNER = "kaggleolap"                 # owner written in the kernel-metadata.json files of the repo
TOKEN_RE = re.compile(r"KGAT_[A-Za-z0-9]+")


def load() -> dict:
    if not os.path.exists(STORE):
        return {"default": None, "accounts": {}}
    with open(STORE, encoding="utf-8") as f:
        return json.load(f)


def save(d: dict):
    tmp = STORE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)
    os.replace(tmp, STORE)


def mask(tok: str) -> str:
    return tok[:5] + "..." + tok[-2:] if len(tok) > 9 else "***"


def env_for(user: str) -> dict:
    d = load()
    if user not in d["accounts"]:
        sys.exit(f"unknown account {user!r}; known: {sorted(d['accounts'])} (add it with `add`)")
    return dict(os.environ, KAGGLE_API_TOKEN=d["accounts"][user]["token"], N5K_KAGGLE_USER=user, PYTHONUTF8="1")


def run(user: str, cmd: list[str], cwd: str | None = None) -> int:
    """Run cmd with the account's token; the token is masked in anything echoed by this script."""
    return subprocess.run(cmd, env=env_for(user), cwd=cwd).returncode


def cmd_add(a):
    text = open(a.token_file, encoding="utf-8").read()
    m = TOKEN_RE.search(text)
    tok = m.group(0) if m else text.strip()
    if not tok or any(c.isspace() for c in tok):
        sys.exit(f"no token found in {a.token_file}")
    d = load()
    d["accounts"][a.user] = {"token": tok}
    d["default"] = d["default"] or a.user
    save(d)
    print(f"stored {a.user} ({mask(tok)}) in {os.path.relpath(STORE, os.getcwd())}")


def cmd_list(a):
    d = load()
    for u, v in sorted(d["accounts"].items()):
        print(f"{'*' if u == d['default'] else ' '} {u:<20} {mask(v['token'])}")
    if not d["accounts"]:
        print("no accounts; add one with `add`")


def cmd_whoami(a):
    sys.exit(run(a.user, ["kaggle", "config", "view"]))


def cmd_run(a):
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    sys.exit(run(a.user, cmd))


def cmd_push_code(a):
    sys.exit(run(a.user, [sys.executable, os.path.join(REPO, "n5k", "kaggle", "push_code.py"), "-m", a.message]))


def cmd_push_kernel(a):
    src = os.path.abspath(a.folder)
    meta_path = os.path.join(src, "kernel-metadata.json")
    meta = json.load(open(meta_path, encoding="utf-8"))
    fix = (lambda s: a.user + s[len(HOME_OWNER):] if s.startswith(HOME_OWNER + "/") else s)
    meta["id"] = fix(meta["id"])
    for k in ("dataset_sources", "kernel_sources"):
        meta[k] = [fix(s) for s in meta.get(k, [])]
    tmp = tempfile.mkdtemp(prefix="kernel_", dir=os.path.dirname(src))
    try:
        shutil.copy(os.path.join(src, meta["code_file"]), tmp)
        json.dump(meta, open(os.path.join(tmp, "kernel-metadata.json"), "w", encoding="utf-8"), indent=1)
        print(f"pushing {meta['id']} (datasets {meta['dataset_sources']}, kernels {meta['kernel_sources']})", flush=True)
        cmd = ["kaggle", "kernels", "push", "-p", os.path.basename(tmp)]
        if not a.cpu:
            cmd += ["--accelerator", "NvidiaTeslaT4"]
        rc = run(a.user, cmd, cwd=os.path.dirname(src))   # relative path: the CLI breaks on absolute Windows paths
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(rc)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="action", required=True)
    p = sub.add_parser("add"); p.add_argument("user"); p.add_argument("--token-file", required=True)
    p.set_defaults(f=cmd_add)
    sub.add_parser("list").set_defaults(f=cmd_list)
    p = sub.add_parser("whoami"); p.add_argument("user"); p.set_defaults(f=cmd_whoami)
    p = sub.add_parser("run"); p.add_argument("user"); p.add_argument("cmd", nargs=argparse.REMAINDER)
    p.set_defaults(f=cmd_run)
    p = sub.add_parser("push-code"); p.add_argument("user"); p.add_argument("-m", "--message", default="update")
    p.set_defaults(f=cmd_push_code)
    p = sub.add_parser("push-kernel"); p.add_argument("user"); p.add_argument("folder")
    p.add_argument("--cpu", action="store_true", help="no GPU (e.g. the run0 cache kernel)")
    p.set_defaults(f=cmd_push_kernel)
    a = ap.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
