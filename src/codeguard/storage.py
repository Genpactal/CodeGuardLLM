"""Single-writer, durable experiment storage with interrupted-tail recovery."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import uuid


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_record(path, record):
    with Path(path).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_records(path, repair=False):
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    result, offset = [], 0
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("Record must be a JSON object")
        except (ValueError, UnicodeDecodeError) as exc:
            if repair and index == len(lines) - 1 and not line.endswith(b"\n"):
                path.with_name(path.name + f".recovered-{uuid.uuid4().hex}").write_bytes(line)
                with path.open("r+b") as handle:
                    handle.truncate(offset)
                break
            raise ValueError(f"Invalid JSONL in {path}, line {index + 1}") from exc
        result.append(record)
        offset += len(line)
    else:
        if repair and data and not data.endswith(b"\n"):
            with path.open("ab") as handle:
                handle.write(b"\n")
    return result


@contextmanager
def run_lock(directory):
    """OS locks release automatically after a crashed process; one host only."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another process is using {directory}") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
