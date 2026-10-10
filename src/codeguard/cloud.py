"""Local work with periodic ZIP checkpoints on a mounted cloud drive."""

from pathlib import Path
import os
import shutil
import time
import zipfile


def extract_archive(archive, destination, root_name):
    """Validate all members before extracting an archive under one named folder."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if (not target.is_relative_to(destination / root_name)
                    or '\\' in member.filename
                    or (member.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError(f'Unsafe archive member: {member.filename}')
        bundle.extractall(destination)


def select_dtype(compute_major, native_bf16):
    # T4 lacks native BF16; FP32 avoids the earlier FP16 sampling failure.
    return 'bfloat16' if compute_major >= 8 and native_bf16 else 'float32'


class Checkpoint:
    """One writer per run; snapshots exclude locks and incomplete JSONL tails."""

    def __init__(self, run_dir, backup, scratch, interval=60):
        self.run_dir = Path(run_dir)
        self.backup = Path(backup) if backup is not None else None
        self.scratch = Path(scratch)
        self.interval = interval
        self.last_saved = time.monotonic()

    def restore(self):
        if self.backup is None or not self.backup.is_file():
            return False
        # Existing local work can be newer than the latest checkpoint.
        if self.run_dir.exists() and any(self.run_dir.iterdir()):
            return False
        extract_archive(self.backup, self.run_dir.parent, self.run_dir.name)
        print(f'Restored checkpoint: {self.backup}', flush=True)
        return True

    def save(self, force=False):
        if self.backup is None or not self.run_dir.exists():
            return
        if not force and time.monotonic() - self.last_saved < self.interval:
            return
        self.scratch.mkdir(parents=True, exist_ok=True)
        local = self.scratch / 'checkpoint.zip'
        with zipfile.ZipFile(local, 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(self.run_dir.rglob('*')):
                if not path.is_file() or path.name == '.lock' or path.suffix == '.tmp':
                    continue
                data = path.read_bytes()
                if path.suffix == '.jsonl':
                    # A subprocess may be appending its next sample during backup.
                    data = data[:data.rfind(b'\n') + 1]
                archive.writestr(path.relative_to(self.run_dir.parent).as_posix(), data)
        self.backup.parent.mkdir(parents=True, exist_ok=True)
        pending = self.backup.with_suffix('.zip.pending')
        shutil.copyfile(local, pending)
        os.replace(pending, self.backup)
        self.last_saved = time.monotonic()
        print(f'Checkpoint saved: {self.backup.name}', flush=True)
