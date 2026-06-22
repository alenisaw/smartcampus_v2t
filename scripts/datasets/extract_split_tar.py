"""Safely extract a gzip tar archive split into alphabetic volume files."""

from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path
from typing import BinaryIO, Sequence


class SplitReader(io.RawIOBase):
    def __init__(self, parts: Sequence[Path]) -> None:
        if not parts:
            raise ValueError("No archive parts supplied")
        self.parts = list(parts)
        self.index = 0
        self.handle: BinaryIO = self.parts[0].open("rb")

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray) -> int:
        total = 0
        view = memoryview(buffer)
        while total < len(buffer) and self.index < len(self.parts):
            count = self.handle.readinto(view[total:])
            if count:
                total += count
                continue
            self.handle.close()
            self.index += 1
            if self.index < len(self.parts):
                self.handle = self.parts[self.index].open("rb")
        return total

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.close()
        super().close()


def _safe_target(destination: Path, member_name: str) -> Path:
    target = (destination / member_name).resolve()
    if target != destination and destination not in target.parents:
        raise RuntimeError(f"Archive member escapes destination: {member_name}")
    return target


def extract(parts: Sequence[Path], destination: Path) -> tuple[int, int]:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    count = 0
    total_bytes = 0
    raw = SplitReader(parts)
    try:
        with io.BufferedReader(raw, buffer_size=4 * 1024 * 1024) as stream:
            with tarfile.open(fileobj=stream, mode="r|gz") as archive:
                for member in archive:
                    _safe_target(destination, member.name)
                    if member.issym() or member.islnk() or member.isdev():
                        raise RuntimeError(f"Unsupported archive link/device: {member.name}")
                    archive.extract(member, path=destination)
                    count += 1
                    total_bytes += member.size if member.isfile() else 0
                    if count % 5000 == 0:
                        print(f"Extracted {count} entries ({total_bytes} bytes)", flush=True)
    except Exception:
        print(f"Extraction failed; partial output retained for inspection: {destination}")
        raise
    return count, total_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", nargs="+", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parts = [path.resolve() for path in args.parts]
    missing = [str(path) for path in parts if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing archive part(s): {', '.join(missing)}")
    count, total_bytes = extract(parts, args.destination)
    print(f"Extraction complete: {count} entries, {total_bytes} file bytes -> {args.destination.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
