"""swebench-pro.sh end to end on the CPU, with stand-ins for docker, vLLM, Harbor and the HEX converter.

What it proves: the checks, the HEX conversion (once, behind a .verified marker), a plain MLX
folder refused, the served model's context length taken from its config, a server on the port
that serves another model left alone, one run folder per model, resume, and the scores files.
"""
import json
import os
import socket
import stat
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("yaml")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "swebench-pro.sh"
IDS = [f"instance_{r}__{r}-{i:03d}" for r in ("a", "b") for i in range(3)]

FAKE_VLLM = r'''#!PY
import json, sys, http.server, os
a = sys.argv[1:]
assert a[0] == "serve"
model, port, name = a[1], int(a[a.index("--port") + 1]), a[a.index("--served-model-name") + 1]
open(os.environ["FAKE_LOG"], "a").write(json.dumps(a) + "\n")
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def _send(self, obj):
        b = json.dumps(obj).encode(); self.send_response(200)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def do_GET(self): self._send({"data": [{"id": name, "root": model}]})
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"])); self._send({"choices": [{"message": {"content": "ok"}}]})
http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
'''

FAKE_HARBOR = r'''#!PY
import json, sys, os
from pathlib import Path
a = sys.argv[1:]
open(os.environ["FAKE_LOG"], "a").write(json.dumps(["harbor"] + a) + "\n")
if a[0] == "job":
    sys.exit(0)
name = a[a.index("--job-name") + 1]
ids = [a[i + 1].lstrip("*") for i, x in enumerate(a) if x == "-i"]
job = Path(a[a.index("-o") + 1]) / name
job.mkdir(parents=True, exist_ok=True)
(job / "config.json").write_text("{}")
for k, iid in enumerate(ids):
    d = job / f"trial{k}"; d.mkdir(exist_ok=True)
    reward = 1.0 if (k % 2 == 0 and "regrade" in name) else 0.0
    (d / "result.json").write_text(json.dumps({"task_name": f"swebench-pro/{iid}",
                                               "verifier_result": {"rewards": {"reward": reward}}}))
'''

FAKE_PIPELINE_PY = r'''#!/bin/sh
# the pipeline venv's python: only `-m blackhole_runtime.verify SRC DST` is used
[ "$1 $2" = "-m blackhole_runtime.verify" ] || exec "PYEXE" "$@"
echo "verify $3 $4" >>"$FAKE_LOG"
mkdir -p "$4"; echo '{"max_position_embeddings": 40960, "quantization_config": {"quant_method": "blackhole-hex"}}' >"$4/config.json"
'''


def _exe(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.replace("#!PY", f"#!{sys.executable}").replace("PYEXE", sys.executable))
    p.chmod(p.stat().st_mode | stat.S_IXUSR)


def _port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def make_vm(tmp_path):
    home, bh, bin_ = tmp_path / "home", tmp_path / "bh", tmp_path / "bin"
    log = tmp_path / "calls.log"
    _exe(bin_ / "docker", "#!/bin/sh\n[ \"$1\" = info ] && [ \"$2\" = -f ] && { echo \"%s\"; exit 0; }\nexit 0\n" % tmp_path)
    _exe(bin_ / "uv", "#!/bin/sh\nexit 0\n")
    _exe(bh / "vllm" / "bin" / "vllm", FAKE_VLLM)
    _exe(bh / "venv" / "bin" / "python", FAKE_PIPELINE_PY)
    sw = bh / "swebench"
    _exe(sw / "venv" / "bin" / "harbor", FAKE_HARBOR)
    (sw / "venv" / "bin" / "python").symlink_to(sys.executable)
    sbp = sw / "SWE-bench_Pro-os" / "v2"
    for i in IDS:
        (sbp / "tasks" / i).mkdir(parents=True)
        (sbp / "tasks" / i / "task.toml").write_text("")
    (sbp / "tooling" / "configs").mkdir(parents=True)
    (sbp / "tooling" / "configs" / "mini_textbased.yaml").write_text("agent: {}\nmodel: {model_kwargs: {}}\n")
    env = {"PATH": f"{bin_}:{os.environ['PATH']}", "HOME": str(home), "BLACKHOLE_HOME": str(bh),
           "FAKE_LOG": str(log), "MIN_START_GB": "0", "USER": "tester"}

    def run(*args, port=None, ok=True):
        p = subprocess.run(["bash", str(SCRIPT), *map(str, args), "--port", str(port or _port()), "-n", "2"],
                           env=env, capture_output=True, text=True, timeout=120)
        if ok:
            assert p.returncode == 0, p.stdout + p.stderr
        return p

    def calls():
        return [json.loads(x) if x.startswith("[") else x for x in log.read_text().splitlines()] if log.exists() else []

    return type("VM", (), {"run": staticmethod(run), "calls": staticmethod(calls), "tmp": tmp_path, "bh": bh, "sw": sw})


@pytest.fixture
def vm(tmp_path):
    return make_vm(tmp_path)


def _model(d: Path, cfg: dict) -> Path:
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(cfg))
    return d


def test_hex_export_converts_once_serves_and_scores(vm):
    m = _model(vm.tmp / "job" / "out" / "m-MLX-9GB", {"hex": {"0": {}}, "quantization": {"group_size": 32}})
    p = vm.run(m, "--tasks", "all", "--name", "j1-swebench-pro")
    assert "3/6 resolved (50.0%)" in p.stdout
    run = vm.sw / "runs" / "j1-swebench-pro"
    res = json.loads((run / "results.json").read_text())
    assert len(res) == 6 and sum(res.values()) == 3
    assert json.loads((run / "results.summary.json").read_text())["resolved_rate"] == 0.5
    conv = Path(f"{m}-vllm")
    assert (conv / ".verified").exists()
    serve = next(c for c in vm.calls() if isinstance(c, list) and c[0] == "serve")
    assert serve[1] == str(conv) and serve[serve.index("--max-model-len") + 1] == "40960"
    assert "--reasoning-parser" not in serve          # unknown family: no parser
    # rerun = resume: no second conversion, Harbor told to resume both jobs
    vm.run(m, "--tasks", "all", "--name", "j1-swebench-pro")
    assert sum(1 for c in vm.calls() if isinstance(c, str) and c.startswith("verify")) == 1
    assert sum(1 for c in vm.calls() if isinstance(c, list) and c[:3] == ["harbor", "job", "resume"]) == 2


def test_plain_mlx_folder_is_refused(vm):
    m = _model(vm.tmp / "m-MLX", {"quantization": {"group_size": 64, "bits": 4}})
    p = vm.run(m, ok=False)
    assert p.returncode != 0 and "without HEX experts" in p.stderr


def test_run_name_is_tied_to_its_model(vm):
    a = _model(vm.tmp / "a", {"max_position_embeddings": 1000000, "model_type": "qwen3_next"})
    b = _model(vm.tmp / "b", {})
    vm.run(a, "--tasks", "2", "--name", "same")
    serve = next(c for c in vm.calls() if isinstance(c, list) and c[0] == "serve")
    assert serve[serve.index("--max-model-len") + 1] == "262144"     # capped
    assert serve[serve.index("--reasoning-parser") + 1] == "qwen3"
    p = vm.run(b, "--tasks", "2", "--name", "same", ok=False)
    assert p.returncode != 0 and "its own --name" in p.stderr


def test_qwen4_family_gets_the_thinking_parser(vm):
    m = _model(vm.tmp / "q4", {"model_type": "qwen4_exp", "text_config": {"model_type": "qwen4_exp_text"}})
    vm.run(m, "--tasks", "2")
    serve = next(c for c in vm.calls() if isinstance(c, list) and c[0] == "serve")
    assert serve[serve.index("--reasoning-parser") + 1] == "qwen3"


def test_other_model_on_the_port_is_not_reused(vm):
    port = _port()
    other = subprocess.Popen([str(vm.bh / "vllm" / "bin" / "vllm"), "serve", "/somewhere/else", "--port", str(port),
                              "--served-model-name", "other"], env={**os.environ, "FAKE_LOG": str(vm.tmp / "other.log")})
    try:
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", port), 0.2).close()
                break
            except OSError:
                import time
                time.sleep(0.1)
        m = _model(vm.tmp / "mine", {})
        p = vm.run(m, "--tasks", "2", port=port)
        assert "port in use by another model" in p.stdout
        serve = next(c for c in vm.calls() if isinstance(c, list) and c[0] == "serve")
        assert serve[1] == str(m) and serve[serve.index("--port") + 1] != str(port)
    finally:
        other.kill()


def test_check_mode_and_disk_floor(vm):
    m = _model(vm.tmp / "m", {})
    p = vm.run(m, "--check")
    assert "checks passed" in p.stdout and not (vm.sw / "runs").exists()
    env_floor = subprocess.run(["bash", str(SCRIPT), str(m), "--check"], capture_output=True, text=True,
                               env={"PATH": f"{vm.tmp / 'bin'}:{os.environ['PATH']}", "HOME": str(vm.tmp / "home"),
                                    "BLACKHOLE_HOME": str(vm.bh), "MIN_START_GB": "999999999", "USER": "tester"})
    assert env_floor.returncode != 0 and "needs at least" in env_floor.stderr
