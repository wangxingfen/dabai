"""就地剥离 web/anim 下所有已下载 FBX 的网格，保留骨架+动画（覆盖原文件）。"""
import json
import os
import subprocess
import sys
import tempfile

BASE = r"D:\AI\dabai\web\anim"
BLENDER = r"D:\blender\blender.exe"
SCRIPT = r"D:\AI\dabai\tools\fbx_strip_skin.py"
CONFIG = os.path.join(BASE, "animation-library.json")

d = json.load(open(CONFIG, encoding="utf-8"))
files = [a["file"] for c in d["categories"].values() for a in c["animations"]]

targets = []
for f in files:
    p = os.path.join(BASE, f.replace("/", os.sep))
    if os.path.exists(p):
        targets.append((f, p))

print("to_strip", len(targets))
ok = fail = 0

def sizeof(p):
    try:
        return int(os.path.getsize(p))
    except Exception:
        return -1

before = {f: sizeof(p) for f, p in targets}
for f, p in targets:
    fd, tmp = tempfile.mkstemp(suffix=".fbx", dir=os.path.dirname(p))
    os.close(fd)
    try:
        r = subprocess.run(
            [BLENDER, "--background", "--factory-startup", "--python", SCRIPT,
             "--", p, tmp],
            capture_output=True, text=True, timeout=240)
        line = [l for l in r.stdout.splitlines() if l.startswith(("OK ", "NO_ARMATURE", "IMPORT_ERR", "EXPORT_ERR", "Traceback"))]
        if r.returncode == 0 and line and line[0].startswith("OK "):
            os.replace(tmp, p)
            ok += 1
            print("STRIPPED", f, sizeof(p) // 1024, "KB")
        else:
            fail += 1
            print("FAIL", f, line[:1])
            if os.path.exists(tmp):
                os.remove(tmp)
    except Exception as e:
        fail += 1
        print("ERR", f, e)
        if os.path.exists(tmp):
            os.remove(tmp)

print("done ok", ok, "fail", fail)