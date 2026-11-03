"""Durable W&B outbox; all SDK/network calls run outside the training process.

Restart an interrupted uploader with:
    python wandb_resilient.py --outbox runs/<run>/wandb_pending/<run_id>
Only the Python standard library is imported by the training-side client.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import secrets
import signal
import shutil
import subprocess
import sys
import time
import traceback


def _windows_kill_on_close_job():
    """Own this process and descendants; kernel closes the job even on a crash.

    The handle deliberately stays open until process exit, is not inheritable,
    and does not allow children to break away. Call before launching the SDK.
    """
    import ctypes
    from ctypes import wintypes as w
    class Basic(ctypes.Structure):
        _fields_ = [("ProcessTime", ctypes.c_longlong), ("JobTime", ctypes.c_longlong),
                    ("Flags", w.DWORD), ("MinWorkingSet", ctypes.c_size_t),
                    ("MaxWorkingSet", ctypes.c_size_t), ("ActiveProcessLimit", w.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", w.DWORD),
                    ("SchedulingClass", w.DWORD)]
    class IO(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in
                    ("ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes", "OtherBytes")]
    class Extended(ctypes.Structure):
        _fields_ = [("Basic", Basic), ("IO", IO), ("ProcessMemory", ctypes.c_size_t),
                    ("JobMemory", ctypes.c_size_t), ("PeakProcessMemory", ctypes.c_size_t),
                    ("PeakJobMemory", ctypes.c_size_t)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
    kernel.CreateJobObjectW.restype = w.HANDLE
    kernel.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
    kernel.SetInformationJobObject.restype = w.BOOL
    kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
    kernel.AssignProcessToJobObject.restype = w.BOOL
    kernel.GetCurrentProcess.restype = w.HANDLE
    kernel.CloseHandle.argtypes = [w.HANDLE]
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = Extended()
    limits.Basic.Flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    try:
        if not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess()):
            raise ctypes.WinError(ctypes.get_last_error())
    except BaseException:
        kernel.CloseHandle(job)
        raise
    return job


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{secrets.token_hex(4)}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def _events(outbox, after=-1, limit=None):
    result = []
    path = Path(outbox) / "events.jsonl"
    if not path.exists():
        return result
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            # A concurrent append, or a killed producer, can leave a partial tail.
            if not line.endswith("\n"):
                break
            event = json.loads(line)
            if event["sequence"] > after:
                result.append(event)
                if limit is not None and len(result) >= limit:
                    break
    return result


@contextmanager
def _worker_lock(outbox):
    """OS-owned lock, automatically released on process exit, on Windows/Linux."""
    stream = (Path(outbox) / "uploader.lock").open("a+b")
    locked = False
    try:
        if os.name == "nt":
            import msvcrt
            if stream.seek(0, os.SEEK_END) == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                locked = True
            except OSError:
                pass
        else:
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except BlockingIOError:
                pass
        yield locked
    finally:
        if locked:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)
        stream.close()


def _process_options():
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {"start_new_session": True}


class BufferedWandb:
    """Local-only training client. Network outages cannot block these methods."""

    def __init__(self, run_dir, *, project, name, config, run_id=None,
                 transmit_seconds=15.0, retry_seconds=60.0, init_timeout=30.0):
        self.id = run_id or secrets.token_hex(4)
        self.outbox = Path(run_dir).resolve() / "wandb_pending" / self.id
        self.outbox.mkdir(parents=True, exist_ok=True)
        previous = _read_json(self.outbox / "run.json", {})
        if previous and previous["project"] != project:
            raise ValueError("A resumed W&B outbox must use its original project")
        _atomic_json(self.outbox / "run.json", {
            "id": self.id, "project": project, "name": name, "config": config,
            "transmit_seconds": transmit_seconds, "retry_seconds": retry_seconds,
            "init_timeout": init_timeout,
        })
        journal = self.outbox / "events.jsonl"
        # Remove only an incomplete final record left by an interrupted writer.
        if journal.exists():
            with journal.open("r+b") as stream:
                content = stream.read()
                if content and not content.endswith(b"\n"):
                    stream.truncate(content.rfind(b"\n") + 1)
        existing = _events(self.outbox)
        self._next_sequence = existing[-1]["sequence"] + 1 if existing else 0
        (self.outbox / "producer_done").unlink(missing_ok=True)
        self._process = None
        self._warned_start = False
        self._start_worker()

    def _start_worker(self):
        if self._process is not None and self._process.poll() is None:
            return
        try:
            # A previous training session may already have an uploader running.
            with _worker_lock(self.outbox) as available:
                if not available:
                    return
            with (self.outbox / "uploader.log").open("a", encoding="utf-8") as log:
                self._process = subprocess.Popen(
                    [sys.executable, "-u", str(Path(__file__).resolve()),
                     "--outbox", str(self.outbox)],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    close_fds=True, **_process_options(),
                )
        except OSError as error:
            if not self._warned_start:
                print(f"W&B uploader unavailable ({error}); local queue retained: {self.outbox}",
                      file=sys.stderr, flush=True)
                self._warned_start = True

    def _append(self, kind, **payload):
        event = {"sequence": self._next_sequence, "kind": kind, **payload}
        with (self.outbox / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._next_sequence += 1
        self._start_worker()

    def log(self, metrics):
        self._append("metrics", data=metrics)

    def queue_artifact(self, path, *, name, metadata, aliases, immutable=False):
        path = Path(path)
        directory = self.outbox / "artifacts"
        directory.mkdir(exist_ok=True)
        snapshot = directory / f"{self._next_sequence:012d}_{path.name}"
        temporary = snapshot.with_suffix(snapshot.suffix + ".tmp")
        try:
            # Trainer publishes checkpoints with atomic replacement, never edits
            # an existing inode. A hard link pins that exact immutable snapshot.
            try:
                if not immutable:
                    raise OSError("Caller did not guarantee atomic checkpoint replacement")
                os.link(path, temporary)
            except OSError:
                shutil.copyfile(path, temporary)
            os.replace(temporary, snapshot)
        finally:
            temporary.unlink(missing_ok=True)
        self._append("artifact", path=str(snapshot.relative_to(self.outbox)),
                     filename=path.name, name=name, metadata=metadata, aliases=aliases)

    def finish(self):
        # Do not join the uploader or call wandb.finish() from training.
        (self.outbox / "producer_done").touch()
        self._start_worker()


def send_once(outbox, sdk=None, batch_size=512):
    """Commit a bounded batch only AFTER the SDK has finished uploading it.

    A failed/interrupted batch stays in the outbox. Stable explicit SDK history
    steps prevent replayed metrics from becoming a second copy in history.
    """
    outbox = Path(outbox)
    settings = _read_json(outbox / "run.json", {})
    cursor = _read_json(outbox / "uploaded.json", {"sequence": -1})["sequence"]
    pending = _events(outbox, after=cursor, limit=batch_size)
    if not pending:
        return 0
    if sdk is None:
        import wandb as sdk
    run = sdk.init(
        project=settings["project"], name=settings["name"], id=settings["id"],
        resume="allow", mode="online", config=settings["config"], allow_val_change=True,
        dir=str(outbox),
        settings=sdk.Settings(init_timeout=settings["init_timeout"], console="off",
                              x_file_stream_transmit_interval=settings["transmit_seconds"]),
    )
    if run is None or getattr(run, "offline", False) is True or getattr(run, "disabled", False) is True:
        raise RuntimeError("W&B did not create an online run; retaining local queue")
    offset_path = outbox / "history_offset.json"
    offset = _read_json(offset_path, None)
    if offset is None:
        offset = {"value": int(run.step)}
        _atomic_json(offset_path, offset)
    next_remote_step = int(run.step)
    run.define_metric("global_step")
    for prefix in ("train/*", "val/*", "val_initial/*", "val_fast/*"):
        run.define_metric(prefix, step_metric="global_step")
    for event in pending:
        if event["kind"] == "metrics":
            history_step = offset["value"] + event["sequence"]
            if history_step >= next_remote_step:
                run.log(event["data"], step=history_step)
        elif event["kind"] == "artifact":
            artifact = sdk.Artifact(event["name"], type="model", metadata=event["metadata"])
            artifact.add_file(str(outbox / event["path"]), name=event["filename"])
            run.log_artifact(artifact, aliases=event["aliases"])
        else:
            raise ValueError(f"Unknown outbox event: {event['kind']}")
    # All blocking SDK work is confined to a disposable child process.
    run.finish()
    _atomic_json(outbox / "uploaded.json", {"sequence": pending[-1]["sequence"]})
    return len(pending)


def _run_upload(command, timeout):
    """Bound a whole SDK process tree, including its background service."""
    if os.name == "nt":
        command = [sys.executable, "-u", str(Path(__file__).resolve()),
                   "--guarded-command", *command]
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, **_process_options())
    try:
        returncode = process.wait(timeout=timeout)
        if returncode:
            raise RuntimeError(f"upload process exited with code {returncode}")
    finally:
        # Clean descendants on SUCCESS, ordinary failure, timeout and interrupt.
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            process.kill()  # On Windows closes the owning Job handle, too.
        process.wait(timeout=10)


def upload_loop(outbox, *, batch_runner=None, sleep=time.sleep, max_cycles=None):
    """Supervisor retries in fresh SDK processes, even after training has ended."""
    outbox = Path(outbox).resolve()
    with _worker_lock(outbox) as acquired:
        if not acquired:
            return
        cycles = 0
        while max_cycles is None or cycles < max_cycles:
            cycles += 1
            settings = _read_json(outbox / "run.json", {})
            cursor = _read_json(outbox / "uploaded.json", {"sequence": -1})["sequence"]
            if not _events(outbox, after=cursor, limit=1):
                _atomic_json(outbox / "status.json", {"state": "caught_up", "time": time.time()})
                if (outbox / "producer_done").exists():
                    return
                sleep(settings["transmit_seconds"])
                continue
            try:
                if batch_runner is None:
                    _run_upload(
                        [sys.executable, "-u", str(Path(__file__).resolve()),
                         "--outbox", str(outbox), "--send-once"],
                        timeout=max(1800, settings["init_timeout"] + 30),
                    )
                else:
                    batch_runner(outbox)
                _atomic_json(outbox / "status.json", {"state": "uploaded_batch", "time": time.time()})
                # Avoid continuously restarting SDK sessions on a busy producer.
                sleep(settings["transmit_seconds"])
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                _atomic_json(outbox / "status.json", {
                    "state": "retrying", "error": message, "time": time.time(),
                    "retry_seconds": settings["retry_seconds"],
                })
                print(f"W&B upload postponed ({message}); retry in {settings['retry_seconds']}s. "
                      "Training is independent of this uploader.", flush=True)
                sleep(settings["retry_seconds"])


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--guarded-command":
        if os.name != "nt":
            raise RuntimeError("Windows guard requested on a non-Windows host")
        job = _windows_kill_on_close_job()
        code = subprocess.call(sys.argv[2:], stdin=subprocess.DEVNULL, **_process_options())
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)  # Kernel closes job and removes any remaining SDK children.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outbox", type=Path, required=True)
    parser.add_argument("--send-once", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.send_once:
        # Failed SDK sessions must not hang in SDK atexit callbacks.
        code = 0
        try:
            send_once(args.outbox)
        except Exception:
            traceback.print_exc()
            code = 1
        finally:
            # run.finish() ends a run; teardown releases the SDK service itself.
            # The outer supervisor bounds this call if the network hangs it.
            sdk = sys.modules.get("wandb")
            teardown = getattr(sdk, "teardown", None)
            if callable(teardown):
                try:
                    teardown()
                except Exception:
                    traceback.print_exc()
                    code = 1
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    upload_loop(args.outbox)


if __name__ == "__main__":
    main()
