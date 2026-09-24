"""Local multi-process launcher (no torchrun rendezvous; works on Windows, CPU/gloo).

    python -m timeprest.dist.launch_local --nproc 2 -m timeprest.dist.train --config ... --set ...
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile


def main():
    args = sys.argv[1:]
    if len(args) < 4 or args[0] != "--nproc" or args[2] != "-m":
        sys.exit(__doc__)
    n, module, rest = int(args[1]), args[3], args[4:]
    fd, init_file = tempfile.mkstemp(prefix="timeprest_init_")
    os.close(fd)
    os.remove(init_file)
    procs = []
    for r in range(n):
        env = dict(os.environ, RANK=str(r), WORLD_SIZE=str(n), LOCAL_RANK=str(r),
                   TIMEPREST_INIT_FILE=init_file)
        procs.append(subprocess.Popen([sys.executable, "-m", module, *rest], env=env))
    import time
    while any(p.poll() is None for p in procs):
        if any(p.poll() not in (None, 0) for p in procs):   # one rank failed: stop the others
            for p in procs:
                if p.poll() is None:
                    p.kill()
        time.sleep(0.2)
    codes = [p.returncode for p in procs]
    if os.path.exists(init_file):
        os.remove(init_file)
    sys.exit(max(codes))


if __name__ == "__main__":
    main()
