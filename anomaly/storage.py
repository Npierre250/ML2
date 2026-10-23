"""
storage.py - one tiny interface, two backends (local folder / HDFS).
The rest of the project never cares which one it is talking to.
"""
import os
import shutil
import subprocess
from pathlib import Path

from anomaly import config


class LocalStorage:
    name = "local"

    def __init__(self, root):
        self.root = Path(root) / "flink"

    def _dir(self, rel):
        p = self.root / rel
        p.mkdir(parents=True, exist_ok=True)
        return p

    def write_text(self, rel, name, text):
        p = self._dir(rel) / name
        tmp = p.with_name("." + p.name + ".tmp")          # hidden temp -> atomic replace
        tmp.write_text(text)
        os.replace(tmp, p)

    def files(self, rel):
        return sorted(f for f in self._dir(rel).iterdir() if f.is_file() and not f.name.startswith("."))

    def read_text(self, rel, name):
        p = self._dir(rel) / name
        return p.read_text() if p.exists() else ""

    def count_files(self, rel):
        return len(self.files(rel))

    def tail(self, rel, n):
        out = []
        for f in reversed(self.files(rel)):
            out = f.read_text().splitlines() + out
            if len(out) >= n:
                break
        return out[-n:]

    def upload(self, local_path, rel):
        shutil.copy(local_path, self._dir(rel) / Path(local_path).name)

    def download(self, rel, name, local_path):
        src = self._dir(rel) / name
        if not src.exists():
            return False
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, local_path)
        return True

    def wipe(self, rel):
        shutil.rmtree(self.root / rel, ignore_errors=True)


class HdfsStorage:
    name = "hdfs"

    def __init__(self, root):
        self.root = root.rstrip("/")

    def _run(self, args, **kw):
        return subprocess.run(["hdfs", "dfs", *args], capture_output=True, **kw)

    def _p(self, rel):
        return f"{self.root}/{rel}"

    def write_text(self, rel, name, text):
        self._run(["-mkdir", "-p", self._p(rel)])
        r = self._run(["-put", "-f", "-", f"{self._p(rel)}/{name}"], input=text.encode())
        if r.returncode != 0:
            raise RuntimeError(f"hdfs put failed: {r.stderr.decode()[:200]}")

    def files(self, rel):
        r = self._run(["-ls", self._p(rel)], text=True)
        names = [l.split()[-1] for l in r.stdout.splitlines() if l.startswith("-")]
        return sorted(names)

    def read_text(self, rel, name):
        r = self._run(["-cat", f"{self._p(rel)}/{name}"], text=True)
        return r.stdout if r.returncode == 0 else ""

    def count_files(self, rel):
        return len(self.files(rel))

    def tail(self, rel, n):
        out = []
        for f in reversed(self.files(rel)):
            r = self._run(["-cat", f], text=True)
            out = r.stdout.splitlines() + out
            if len(out) >= n:
                break
        return out[-n:]

    def upload(self, local_path, rel):
        self._run(["-mkdir", "-p", self._p(rel)])
        self._run(["-put", "-f", str(local_path), self._p(rel) + "/"])

    def download(self, rel, name, local_path):
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        return self._run(["-get", "-f", f"{self._p(rel)}/{name}", str(local_path)]).returncode == 0

    def wipe(self, rel):
        self._run(["-rm", "-r", "-f", self._p(rel)])


def get_storage():
    if config.STORAGE == "hdfs":
        return HdfsStorage(config.HDFS_ROOT)
    return LocalStorage(config.LOCAL_STORE)
