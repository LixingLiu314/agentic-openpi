"""Configuration and process helpers shared by the small Piper GUI and its tests."""
import errno
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from piper_checkpoint import checkpoint_info, require_native_policy_contract


DEFAULT_TASKS = [
    {"name": "茄子放入盒子", "prompt": "Put the eggplant into the box"},
    {"name": "红薯放入盒子", "prompt": "Put the sweet potato into the box"},
]
DEFAULT_CHECKPOINT = "checkpoints/pi05_piper_stage1/m3_pilot_seed42/step_003500"


def resolve_checkpoint(root, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(root) / path).resolve()


def discover_checkpoints(root):
    results = []
    for metadata in (Path(root) / "checkpoints").glob("*/*/*/metadata.json"):
        try:
            results.append(checkpoint_info(metadata.parent, root=root))
        except (ValueError, KeyError, OSError):
            continue
    return sorted(results, key=lambda item: item["path"])


class Settings:
    def __init__(self, path):
        self.path = Path(path).expanduser()
        self.data = {"tasks": [dict(item) for item in DEFAULT_TASKS], "checkpoints": []}
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                self.data.update(saved)
        except (OSError, ValueError):
            pass
        tasks = self.data.get("tasks")
        if not isinstance(tasks, list) or not all(isinstance(t, dict) and isinstance(t.get("name"), str)
                                                 and isinstance(t.get("prompt"), str) for t in tasks):
            self.data["tasks"] = [dict(item) for item in DEFAULT_TASKS]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def save_task(self, name, prompt):
        name, prompt = name.strip(), prompt.strip()
        if not name or not prompt:
            raise ValueError("请填写任务名称和任务指令")
        tasks = self.data["tasks"]
        for item in tasks:
            if item["name"] == name:
                item["prompt"] = prompt
                break
        else:
            tasks.append({"name": name, "prompt": prompt})
        self.save()


def client_command(root, config, output, telemetry_port, token, kind="inference"):
    if kind not in ("inference", "check", "reset"):
        raise ValueError("Unknown run kind")
    prompt = config["prompt"].strip()
    if kind == "inference" and not prompt:
        raise ValueError("任务指令不能为空")
    if not 1 <= config["chunk_steps"] <= 50 or config["max_steps"] < 1:
        raise ValueError("Invalid step budget")
    command = ["bash", str(Path(root)/"scripts/run_m3_eggplant.sh"),
               "--host", config["host"], "--port", str(config["port"]),
               "--prompt", prompt, "--max-steps", str(config["max_steps"]),
               "--chunk-steps", str(config["chunk_steps"]), "--output", str(output),
               "--telemetry-port", str(telemetry_port), "--telemetry-token", token]
    if kind == "inference":
        command.append("--rtc" if config.get("rtc", True) else "--no-rtc")
    if kind == "check":
        command += ["--check-only"]
    elif kind == "reset":
        command += ["--execute", "--reset-only"]
    else:
        command += ["--execute", "--reset-before" if config["reset"] else "--no-reset"]
    if kind == "inference":
        command += ["--expected-checkpoint", config["checkpoint"]]
    if kind == "reset" or not config["record"]:
        command.append("--no-record-continuous")
    return command


def server_metadata(host, port):
    from openpi_client import msgpack_numpy
    from websockets.sync.client import connect
    with connect("ws://%s:%d" % (host, port), compression=None, max_size=None, open_timeout=2) as connection:
        return msgpack_numpy.unpackb(connection.recv(timeout=2))


def validate_server(metadata, root, checkpoint):
    require_native_policy_contract(metadata)
    advertised = metadata.get("checkpoint")
    if not advertised or resolve_checkpoint(root, advertised) != resolve_checkpoint(root, checkpoint):
        raise ValueError("当前端口加载的 checkpoint 与选择不一致，请先停止该模型服务再加载")
    info = checkpoint_info(resolve_checkpoint(root, checkpoint), root=root)
    if info["kind"] in ("backbone_grad", "official_backbone_grad", "recurrent_subtask"):
        for key in ("stage", "variant", "mode", "weights_sha256", "parent_weights_sha256", "experimental"):
            if metadata.get(key) != info[key]:
                raise ValueError("Backbone 服务身份与选择不一致：" + key)
        if info["kind"] in ("official_backbone_grad", "recurrent_subtask"):
            for key in ("initialization", "inherited_training_updates"):
                if metadata.get(key) != info[key]:
                    raise ValueError("Official 服务初始化身份与选择不一致：" + key)
        if info["kind"] == "recurrent_subtask" and metadata.get("arm") != info["arm"]:
            raise ValueError("服务加载的 S 实验与选择不一致")
        if "experiment" in info and metadata.get("experiment") != info["experiment"]:
            raise ValueError("服务加载的语义实验与选择不一致")
        if "label_version" in info and metadata.get("label_version") != info["label_version"]:
            raise ValueError("服务加载的 reach 标签版本不一致")
    elif info["kind"] == "r1":
        for key in ("variant", "decoder_sha256", "parent_weights_sha256", "experimental"):
            if metadata.get(key) != info[key]:
                raise ValueError("R1 服务身份与选择不一致：" + key)
        if resolve_checkpoint(root, metadata.get("parent_checkpoint") or "") != Path(info["parent_checkpoint"]):
            raise ValueError("R1 服务的 M3 基座与选择不一致")
    elif metadata.get("stage") != "m3" or metadata.get("variant") is not None:
        raise ValueError("选择的是 M3，但服务加载了其他模型变体")
    return metadata


def process_record(pid):
    proc = Path("/proc") / str(pid)
    return {"pid": pid, "start_ticks": proc.joinpath("stat").read_text().split(") ", 1)[1].split()[19]}


def alive(record):
    try:
        return process_record(record["pid"])["start_ticks"] == record["start_ticks"]
    except (OSError, KeyError, IndexError):
        return False


def project_processes(root, script):
    root = Path(root).resolve()
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid() or proc.joinpath("cwd").resolve() != root:
                continue
            arguments = proc.joinpath("cmdline").read_bytes().decode().strip("\0").split("\0")
            if any(arg in ("scripts/" + script, str(root / "scripts" / script)) for arg in arguments):
                found.append(dict(process_record(int(proc.name)), arguments=arguments))
        except (OSError, ValueError, IndexError):
            pass
    return found


MODEL_SCRIPTS = ("serve_subtask_policy.py", "serve_official_gradient_piper.py",
                 "serve_backbone_gradient_piper.py", "serve_recurrent_subtask_piper.py")
STOP_STAGES = ((signal.SIGINT, 15), (signal.SIGTERM, 5), (signal.SIGKILL, 2))


def model_processes(root):
    return [record for script in MODEL_SCRIPTS for record in project_processes(root, script)]


def server_port(record):
    # Wrappers put defaults first; argparse uses the final effective value.
    arguments = record["arguments"]
    port = 8000
    for index, argument in enumerate(arguments):
        if argument == "--port" and index + 1 < len(arguments):
            port = int(arguments[index + 1])
        elif argument.startswith("--port="):
            port = int(argument.split("=", 1)[1])
    return port


def find_server(root, port):
    return next((record for record in model_processes(root) if server_port(record) == int(port)), None)


def port_available(port):
    # Match asyncio's Unix SO_REUSEADDR behavior: TIME_WAIT is not a listener.
    # listen() also catches another process that bound with SO_REUSEADDR.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", int(port)))
            probe.listen(1)
        except OSError as error:
            if error.errno == errno.EADDRINUSE:
                return False
            raise
    return True


def wait_port_release(port, timeout=5):
    deadline = time.monotonic() + timeout
    while not port_available(port):
        if time.monotonic() >= deadline:
            raise RuntimeError("端口 %d 仍被占用，未启动新模型；请检查其他服务或选择空闲端口" % port)
        time.sleep(0.1)


def server_running(record):
    if not alive(record):
        return False
    try:
        return Path("/proc/%d/stat" % record["pid"]).read_text().split(") ", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def stop_server(root, record):
    if project_processes(root, "run_subtask_piper.py"):
        raise RuntimeError("仍有推理客户端运行，请先停止该轮推理")
    current = {item["pid"]: item for item in model_processes(root)}
    owned = current.get(record["pid"])
    if not alive(record) or owned is None or owned.get("start_ticks") != record.get("start_ticks"):
        raise RuntimeError("模型服务进程身份已变化，未发送停止信号")
    port = server_port(owned)
    for sig, timeout in STOP_STAGES:
        if not server_running(record):
            break
        # Recheck ownership and active clients before every escalation.
        current = {item["pid"]: item for item in model_processes(root)}
        if current.get(record["pid"], {}).get("start_ticks") != record["start_ticks"] or not alive(record):
            break
        if project_processes(root, "run_subtask_piper.py"):
            raise RuntimeError("仍有推理客户端运行，请先停止该轮推理")
        try:
            os.kill(record["pid"], sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + timeout
        while server_running(record) and time.monotonic() < deadline:
            time.sleep(0.1)
    if server_running(record):
        raise RuntimeError("模型尚未退出；未启动新模型，请查看服务日志")
    wait_port_release(port)


def clear_model_port(root, port):
    # Only positively identified model processes in this checkout are stoppable.
    record = find_server(root, port)
    if record is not None:
        stop_server(root, record)
    wait_port_release(port)


def server_command(root, checkpoint, port):
    info = checkpoint_info(resolve_checkpoint(root, checkpoint), root=root)
    command = ["bash", str(Path(root)/"scripts/serve_m3_piper.sh"),
               "--checkpoint", info["path"], "--port", str(port)]
    if info["kind"] == "r1":
        command += ["--parent-checkpoint", info["parent_checkpoint"]]
        if info["experimental"]:
            command.append("--allow-unqualified-r1")
    return command


def launch_server(root, checkpoint, port):
    command = server_command(root, checkpoint, port)
    clear_model_port(root, port)
    output = Path(root)/"logs/gui"
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / ("server_%s_%d.log" % (time.strftime("%Y%m%d_%H%M%S"), time.time_ns() % 1000000))
    with log_path.open("ab") as log:
        process = subprocess.Popen(command, cwd=str(root), stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    return process, log_path
