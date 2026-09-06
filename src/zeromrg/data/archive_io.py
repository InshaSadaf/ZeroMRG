"""Read-only access to dataset folders or ZIP archives."""

from __future__ import annotations

import hashlib
import io
import json
import struct
import zlib
import zipfile
from collections import defaultdict
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable


class DatasetSourceError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceMember:
    path: str
    size: int
    crc32: int | None


@dataclass(frozen=True)
class ImageInspection:
    path: str
    size: int
    format: str | None
    width: int | None
    height: int | None
    mode: str | None
    channels: int | None
    corrupt: bool
    error: str | None
    content_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "source_path": self.path,
            "size_bytes": self.size,
            "format": self.format,
            "width": self.width,
            "height": self.height,
            "mode": self.mode,
            "channels": self.channels,
            "corrupt": self.corrupt,
            "error": self.error,
            "content_sha256": self.content_sha256,
        }


class DatasetSource(AbstractContextManager["DatasetSource"]):
    """Uniform immutable member access for one folder or ZIP file."""

    def __init__(self, path: Path, members: list[SourceMember], archive: zipfile.ZipFile | None):
        self.path = path
        self.members = tuple(sorted(members, key=lambda item: item.path))
        self._archive = archive
        self.kind = "zip" if archive is not None else "directory"
        self._index = {member.path: member for member in self.members}

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._archive is not None:
            self._archive.close()

    def open(self, relative_path: str) -> BinaryIO:
        if relative_path not in self._index:
            raise DatasetSourceError(f"Unknown source member: {relative_path}")
        if self._archive is not None:
            return self._archive.open(relative_path, "r")
        return (self.path / Path(PurePosixPath(relative_path))).open("rb")

    def read_bytes(self, relative_path: str) -> bytes:
        with self.open(relative_path) as handle:
            return handle.read()

    def find_by_basename(self, basename: str) -> list[str]:
        return [member.path for member in self.members if PurePosixPath(member.path).name == basename]

    def inventory_hash(self, paths: Iterable[str]) -> str:
        inventory = []
        for path in sorted(paths):
            member = self._index[path]
            inventory.append({"path": path, "size": member.size, "crc32": member.crc32})
        payload = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _directory_members(path: Path) -> list[SourceMember]:
    return [
        SourceMember(file.relative_to(path).as_posix(), file.stat().st_size, None)
        for file in path.rglob("*")
        if file.is_file()
    ]


def _zip_members(archive: zipfile.ZipFile) -> list[SourceMember]:
    return [
        SourceMember(info.filename, info.file_size, info.CRC)
        for info in archive.infolist()
        if not info.is_dir()
    ]


def open_dataset_source(
    configured_path: str | Path, required_basenames: Iterable[str]
) -> DatasetSource:
    """Open a configured ZIP/folder, locating one nested ZIP when necessary."""

    path = Path(configured_path).expanduser().resolve()
    if not path.exists():
        raise DatasetSourceError(f"Configured dataset path does not exist: {path}")
    required = set(required_basenames)

    if path.is_file():
        if not zipfile.is_zipfile(path):
            raise DatasetSourceError(f"Dataset file is not a readable ZIP archive: {path}")
        archive = zipfile.ZipFile(path, "r")
        source = DatasetSource(path, _zip_members(archive), archive)
        available = {PurePosixPath(member.path).name for member in source.members}
        if not required.issubset(available):
            source.__exit__(None, None, None)
            raise DatasetSourceError(
                f"Dataset archive {path} is missing required files: {sorted(required - available)}"
            )
        return source

    members = _directory_members(path)
    available = {PurePosixPath(member.path).name for member in members}
    if required.issubset(available):
        return DatasetSource(path, members, None)

    matching_archives: list[Path] = []
    for candidate in sorted(path.rglob("*.zip")):
        try:
            with zipfile.ZipFile(candidate, "r") as archive:
                names = {PurePosixPath(info.filename).name for info in archive.infolist()}
                if required.issubset(names):
                    matching_archives.append(candidate)
        except zipfile.BadZipFile:
            continue
    if len(matching_archives) == 1:
        return open_dataset_source(matching_archives[0], required)
    if len(matching_archives) > 1:
        raise DatasetSourceError(
            "Multiple nested archives match the required metadata; configure the intended ZIP directly: "
            + ", ".join(str(item) for item in matching_archives)
        )
    raise DatasetSourceError(
        f"Configured directory {path} does not contain required files {sorted(required)}"
    )


def unique_member_by_basename(source: DatasetSource, basename: str) -> str:
    matches = source.find_by_basename(basename)
    if len(matches) != 1:
        raise DatasetSourceError(
            f"Expected exactly one {basename!r}; found {len(matches)}: {matches[:10]}"
        )
    return matches[0]


def _parse_png(data: bytes) -> tuple[str, int, int, str, int]:
    if len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("invalid PNG signature/header")
    ihdr_length = struct.unpack(">I", data[8:12])[0]
    if ihdr_length != 13 or data[12:16] != b"IHDR":
        raise ValueError("missing PNG IHDR")
    width, height, bit_depth, color_type = struct.unpack(">IIBB", data[16:26])
    modes = {0: ("L", 1), 2: ("RGB", 3), 3: ("P", 1), 4: ("LA", 2), 6: ("RGBA", 4)}
    if color_type not in modes or width <= 0 or height <= 0:
        raise ValueError("unsupported or invalid PNG IHDR")
    offset = 8
    found_iend = False
    while offset + 12 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        end = offset + 12 + length
        if end > len(data):
            raise ValueError("truncated PNG chunk")
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", data[offset + 8 + length : end])[0]
        actual_crc = zlib.crc32(chunk_type)
        actual_crc = zlib.crc32(chunk_data, actual_crc) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ValueError("PNG chunk CRC mismatch")
        offset = end
        if chunk_type == b"IEND":
            found_iend = True
            break
    if not found_iend:
        raise ValueError("PNG IEND missing")
    mode, channels = modes[color_type]
    if bit_depth != 8:
        mode = f"{mode};{bit_depth}"
    return "PNG", width, height, mode, channels


def _parse_jpeg(data: bytes) -> tuple[str, int, int, str, int]:
    if len(data) < 4 or data[:2] != b"\xff\xd8" or data[-2:] != b"\xff\xd9":
        raise ValueError("invalid or truncated JPEG")
    offset = 2
    sof_markers = set(range(0xC0, 0xC4)) | set(range(0xC5, 0xC8)) | set(range(0xC9, 0xCC)) | set(range(0xCD, 0xD0))
    while offset + 4 <= len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker in {0x01, *range(0xD0, 0xD9)}:
            continue
        if offset + 2 > len(data):
            break
        length = struct.unpack(">H", data[offset : offset + 2])[0]
        if length < 2 or offset + length > len(data):
            raise ValueError("invalid JPEG segment length")
        if marker in sof_markers:
            if length < 8:
                raise ValueError("invalid JPEG SOF")
            height, width, components = struct.unpack(">HHB", data[offset + 3 : offset + 8])
            if width <= 0 or height <= 0 or components <= 0:
                raise ValueError("invalid JPEG dimensions/components")
            mode = {1: "L", 3: "RGB", 4: "CMYK"}.get(components, f"{components}-component")
            return "JPEG", width, height, mode, components
        offset += length
    raise ValueError("JPEG SOF marker missing")


def inspect_images(source: DatasetSource, image_paths: Iterable[str]) -> dict[str, ImageInspection]:
    """Inspect sequentially and hash only plausible exact-duplicate candidates."""

    paths = sorted(set(image_paths))
    members = {member.path: member for member in source.members}
    buckets: dict[tuple[int, int | None], list[str]] = defaultdict(list)
    for path in paths:
        member = members[path]
        key = (member.size, member.crc32 if source.kind == "zip" else None)
        buckets[key].append(path)
    hash_candidates = {path for group in buckets.values() if len(group) > 1 for path in group}

    results: dict[str, ImageInspection] = {}
    for path in paths:
        member = members[path]
        content_hash = None
        try:
            data = source.read_bytes(path)
            if len(data) != member.size:
                raise ValueError(f"size mismatch: expected {member.size}, read {len(data)}")
            if path in hash_candidates:
                content_hash = hashlib.sha256(data).hexdigest()
            if data.startswith(b"\x89PNG"):
                image_format, width, height, mode, channels = _parse_png(data)
            elif data.startswith(b"\xff\xd8"):
                image_format, width, height, mode, channels = _parse_jpeg(data)
            else:
                raise ValueError("unrecognized image signature")
            result = ImageInspection(
                path, member.size, image_format, width, height, mode, channels, False, None, content_hash
            )
        except (OSError, ValueError, zipfile.BadZipFile, zlib.error) as exc:
            result = ImageInspection(
                path, member.size, None, None, None, None, None, True, str(exc), content_hash
            )
        results[path] = result
    return results


def duplicate_content_groups(inspections: Iterable[ImageInspection]) -> list[list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for inspection in inspections:
        if inspection.content_sha256:
            groups[inspection.content_sha256].append(inspection.path)
    return [sorted(paths) for paths in groups.values() if len(paths) > 1]

