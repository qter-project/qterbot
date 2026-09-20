"""Load cached DiscordChatExporter tar archives into Polars dataframes.

The archive directory contains a ``media`` directory, numbered ``.tar.gz``
exports, and optional per-frame Parquet caches.  ``load_dataframes`` loads
all of the data into dataframes while caching the processed data in the parquet
files for faster future access.
"""

from __future__ import annotations

import json
import os
import re
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from uuid import uuid4

import polars as pl

PROJECT_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = PROJECT_DIR / "qter-channels"
NAME_MAPPINGS_FILE = PROJECT_DIR / "username-real-names.txt"


MESSAGE_SCHEMA: dict[str, pl.DataType] = {
    "author_id": pl.String,
    "content": pl.String,
    "timestamp": pl.Datetime(time_zone="UTC"),
    "message_id": pl.String,
    "guild": pl.String,
    "category": pl.String,
    "channel": pl.String,
}
USER_SCHEMA: dict[str, pl.DataType] = {
    "user_id": pl.String,
    "usernames": pl.List(pl.String),
    "nicknames": pl.List(pl.String),
    "full_real_name": pl.String,
}
USER_CACHE_SCHEMA = {
    key: value for key, value in USER_SCHEMA.items() if key != "full_real_name"
}
CHANNEL_SCHEMA: dict[str, pl.DataType] = {
    "channel_name": pl.String,
    "channel_id": pl.String,
    "category_name": pl.String,
    "category_id": pl.String,
    "guild_name": pl.String,
    "guild_id": pl.String,
}

FRAME_SCHEMAS = {
    "messages": MESSAGE_SCHEMA,
    "users": USER_CACHE_SCHEMA,
    "channels": CHANNEL_SCHEMA,
}
EXPORT_NAME = re.compile(r"^(?P<index>\d+)(?:-(?P<label>.*))?\.tar\.gz$")
CACHE_NAME = re.compile(
    r"^(?P<index>\d+)-(?P<frame>messages|users|channels)\.parquet$"
)


@dataclass(frozen=True)
class ArchiveDataFrames:
    messages: pl.DataFrame
    users: pl.DataFrame
    channels: pl.DataFrame


@dataclass(frozen=True)
class _ArchiveSnapshot:
    exports: dict[int, Path]
    caches: dict[str, dict[int, Path]]


def _read_real_name_mappings(path: Path) -> dict[str, str]:
    """Read ``username -> Full Real Name`` mappings, ignoring comments."""
    if not path.exists():
        return {}

    mappings: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "->" not in line:
            raise ValueError(
                f"{path}:{line_number}: expected 'username -> Full Real Name'"
            )
        username, full_name = (
            part.strip() for part in line.split("->", maxsplit=1)
        )
        if not username or not full_name:
            raise ValueError(
                f"{path}:{line_number}: username and full real name are both required"
            )
        mappings.setdefault(username.casefold(), full_name)
    return mappings


def _empty_frame(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


def _archive_snapshot(archive_dir: Path) -> _ArchiveSnapshot:
    """List the recognized archive artifacts once and validate export indices."""
    if not archive_dir.is_dir():
        raise FileNotFoundError(
            f"archive directory does not exist: {archive_dir}"
        )
    media_dir = archive_dir / "media"
    if not media_dir.is_dir():
        raise FileNotFoundError(
            f"archive directory is missing media directory: {media_dir}"
        )

    exports: dict[int, Path] = {}
    caches = {frame: {} for frame in FRAME_SCHEMAS}
    for path in archive_dir.iterdir():
        export_match = EXPORT_NAME.fullmatch(path.name)
        cache_match = CACHE_NAME.fullmatch(path.name)
        if export_match:
            if not path.is_file():
                raise ValueError(f"export is not a regular file: {path}")
            index = int(export_match["index"])
            if index in exports:
                raise ValueError(
                    f"multiple exports use index {index}: {exports[index]}, {path}"
                )
            exports[index] = path
        elif cache_match:
            if not path.is_file():
                raise ValueError(
                    f"Parquet cache is not a regular file: {path}"
                )
            index = int(cache_match["index"])
            frame = cache_match["frame"]
            if index in caches[frame]:
                raise ValueError(
                    f"multiple {frame} cache files use index {index}: "
                    f"{caches[frame][index]}, {path}"
                )
            caches[frame][index] = path

    if exports:
        expected = set(range(max(exports) + 1))
        missing = sorted(expected - exports.keys())
        if missing:
            raise ValueError(
                f"archive exports must be contiguous from 0; missing indices: {missing}"
            )
        latest_export = max(exports)
        cache_ahead = [
            path
            for frame_caches in caches.values()
            for index, path in frame_caches.items()
            if index > latest_export
        ]
        if cache_ahead:
            raise ValueError(
                f"Parquet cache refers to a missing export: {cache_ahead[0]}"
            )
    elif any(caches.values()):
        raise ValueError(
            "Parquet caches exist but the archive contains no exports"
        )
    return _ArchiveSnapshot(exports=exports, caches=caches)


def _parse_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _string(value: Any) -> str | None:
    return None if value is None else str(value)


def _required_string(value: Any, description: str, source: str) -> str:
    result = _string(value)
    if result is None:
        raise ValueError(f"{source}: missing {description}")
    return result


def _validate_member(member: tarfile.TarInfo, archive_path: Path) -> None:
    member_path = PurePosixPath(member.name)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise ValueError(
            f"{archive_path}: unsafe tar member path: {member.name}"
        )
    if member.isdir():
        return
    if not member.isfile():
        raise ValueError(
            f"{archive_path}: non-regular tar member: {member.name}"
        )
    if not member.name.endswith(".json"):
        raise ValueError(f"{archive_path}: non-JSON tar member: {member.name}")


def _exports_from_tarball(
    archive_path: Path,
) -> Iterable[tuple[str, dict[str, Any]]]:
    """Yield validated JSON export objects in their tar-member order."""
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive:
                _validate_member(member, archive_path)
                if member.isdir():
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(
                        f"{archive_path}: could not read tar member {member.name}"
                    )
                source = f"{archive_path}:{member.name}"
                try:
                    export = json.load(extracted)
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    raise ValueError(f"{source}: invalid JSON") from error
                if not isinstance(export, dict):
                    raise ValueError(
                        f"{source}: export JSON must be an object"
                    )
                yield source, export
    except (tarfile.TarError, OSError) as error:
        raise ValueError(f"{archive_path}: invalid tar.gz export") from error


def _frames_from_exports(
    exports: Iterable[tuple[str, dict[str, Any]]],
) -> ArchiveDataFrames:
    """Adapt the original JSON-export parser for an iterable of tar members."""
    message_rows: list[dict[str, Any]] = []
    user_rows: list[dict[str, str | None]] = []
    channel_rows: list[dict[str, str | None]] = []

    for source, export in exports:
        guild = export.get("guild")
        channel = export.get("channel")
        messages = export.get("messages")
        if not isinstance(guild, dict):
            raise ValueError(f"{source}: guild must be an object")
        if not isinstance(channel, dict):
            raise ValueError(f"{source}: channel must be an object")
        if not isinstance(messages, list):
            raise ValueError(f"{source}: messages must be a list")

        channel_id = _required_string(channel.get("id"), "channel id", source)
        guild_id = _required_string(guild.get("id"), "guild id", source)
        channel_rows.append(
            {
                "channel_name": _string(channel.get("name")),
                "channel_id": channel_id,
                "category_name": _string(channel.get("category")),
                "category_id": _string(channel.get("categoryId")),
                "guild_name": _string(guild.get("name")),
                "guild_id": guild_id,
            }
        )

        for position, message in enumerate(messages):
            message_source = f"{source}:messages[{position}]"
            if not isinstance(message, dict):
                raise ValueError(
                    f"{message_source}: message must be an object"
                )
            author = message.get("author")
            if not isinstance(author, dict):
                raise ValueError(f"{message_source}: author must be an object")
            author_id = _required_string(
                author.get("id"), "author id", message_source
            )
            message_rows.append(
                {
                    "author_id": author_id,
                    "content": _string(message.get("content")),
                    "timestamp": _parse_timestamp(message.get("timestamp")),
                    "message_id": _required_string(
                        message.get("id"), "message id", message_source
                    ),
                    "guild": _string(guild.get("name")),
                    "category": _string(channel.get("category")),
                    "channel": channel_id,
                }
            )
            user_rows.append(
                {
                    "user_id": author_id,
                    "username": _string(author.get("name")),
                    "nickname": _string(author.get("nickname")),
                }
            )

    messages = (
        _empty_frame(MESSAGE_SCHEMA)
        if not message_rows
        else pl.DataFrame(message_rows, schema=MESSAGE_SCHEMA)
    )
    users_source = (
        _empty_frame(
            {
                "user_id": pl.String,
                "username": pl.String,
                "nickname": pl.String,
            }
        )
        if not user_rows
        else pl.DataFrame(
            user_rows,
            schema={
                "user_id": pl.String,
                "username": pl.String,
                "nickname": pl.String,
            },
        )
    )
    channels_source = (
        _empty_frame(CHANNEL_SCHEMA)
        if not channel_rows
        else pl.DataFrame(channel_rows, schema=CHANNEL_SCHEMA)
    )

    users = (
        users_source.lazy()
        .group_by("user_id", maintain_order=True)
        .agg(
            pl.col("username")
            .drop_nulls()
            .unique(maintain_order=True)
            .alias("usernames"),
            pl.col("nickname")
            .drop_nulls()
            .unique(maintain_order=True)
            .alias("nicknames"),
        )
        .select(*USER_CACHE_SCHEMA)
        .collect()
    )
    channels = (
        channels_source.lazy()
        .unique(
            subset=["channel_id", "guild_id"], keep="last", maintain_order=True
        )
        .select(*CHANNEL_SCHEMA)
        .collect()
    )
    return ArchiveDataFrames(messages=messages, users=users, channels=channels)


def _read_latest_cache(
    candidates: dict[int, Path], frame: str
) -> tuple[int, pl.DataFrame]:
    """Read the newest schema-valid cache for one frame, or an empty frame."""
    schema = FRAME_SCHEMAS[frame]
    for index in sorted(candidates, reverse=True):
        try:
            cached = pl.read_parquet(candidates[index])
        except Exception:
            continue
        if cached.schema == schema:
            return index, cached
    return -1, _empty_frame(schema)


def _merge_messages(cached: pl.DataFrame, fresh: pl.DataFrame) -> pl.DataFrame:
    return (
        pl.concat([cached, fresh], how="vertical")
        .lazy()
        .unique(subset=["message_id"], keep="last", maintain_order=True)
        .select(*MESSAGE_SCHEMA)
        .collect()
    )


def _merge_users(cached: pl.DataFrame, fresh: pl.DataFrame) -> pl.DataFrame:
    return (
        pl.concat([cached, fresh], how="vertical")
        .lazy()
        .group_by("user_id", maintain_order=True)
        .agg(
            pl.col("usernames")
            .explode(empty_as_null=True)
            .drop_nulls()
            .unique(maintain_order=True)
            .alias("usernames"),
            pl.col("nicknames")
            .explode(empty_as_null=True)
            .drop_nulls()
            .unique(maintain_order=True)
            .alias("nicknames"),
        )
        .select(*USER_CACHE_SCHEMA)
        .collect()
    )


def _merge_channels(cached: pl.DataFrame, fresh: pl.DataFrame) -> pl.DataFrame:
    return (
        pl.concat([cached, fresh], how="vertical")
        .lazy()
        .unique(
            subset=["channel_id", "guild_id"], keep="last", maintain_order=True
        )
        .select(*CHANNEL_SCHEMA)
        .collect()
    )


def _with_real_names(
    users: pl.DataFrame, mappings: dict[str, str]
) -> pl.DataFrame:
    full_names = [
        next(
            (
                mappings[username.casefold()]
                for username in usernames
                if username.casefold() in mappings
            ),
            None,
        )
        for usernames in users.get_column("usernames").to_list()
    ]
    return users.with_columns(
        pl.Series("full_real_name", full_names, dtype=pl.String)
    ).select(*USER_SCHEMA)


def _new_export_records(
    snapshot: _ArchiveSnapshot, after_index: int
) -> list[tuple[int, str, dict[str, Any]]]:
    """Read each export needed by any frame once, preserving source order."""
    records: list[tuple[int, str, dict[str, Any]]] = []
    for export_index in range(after_index + 1, len(snapshot.exports)):
        records.extend(
            (export_index, source, export)
            for source, export in _exports_from_tarball(
                snapshot.exports[export_index]
            )
        )
    return records


def _records_after(
    records: Iterable[tuple[int, str, dict[str, Any]]], index: int
) -> Iterable[tuple[str, dict[str, Any]]]:
    return (
        (source, export)
        for export_index, source, export in records
        if export_index > index
    )


def _write_caches(
    archive_dir: Path,
    index: int,
    frames: ArchiveDataFrames,
    initial_caches: dict[str, dict[int, Path]],
) -> None:
    """Publish each frame through a unique temporary filename, then clean old sets."""
    cache_frames = {
        "messages": frames.messages,
        "users": frames.users.select(*USER_CACHE_SCHEMA),
        "channels": frames.channels,
    }
    for frame, dataframe in cache_frames.items():
        final_path = archive_dir / f"{index}-{frame}.parquet"
        with tempfile.NamedTemporaryFile(
            dir=archive_dir,
            prefix=f".{index}-{frame}-{uuid4().hex}-",
            suffix=".parquet.tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        dataframe.write_parquet(temporary_path)
        os.replace(temporary_path, final_path)

    # Only remove cache triplets observed before this call.  Orphaned files may
    # belong to a concurrent writer or an interrupted prior update, so leave them.
    complete_indices = set.intersection(
        *(set(caches) for caches in initial_caches.values())
    )
    for old_index in complete_indices - {index}:
        for frame in FRAME_SCHEMAS:
            initial_caches[frame][old_index].unlink(missing_ok=True)


def load_dataframes(
    archive_dir: Path = ARCHIVE_DIR,
    name_mappings_file: Path = NAME_MAPPINGS_FILE,
) -> ArchiveDataFrames:
    """Load all numbered tar exports into cached messages, users, and channels frames."""
    snapshot = _archive_snapshot(archive_dir)
    mappings = _read_real_name_mappings(name_mappings_file)
    if not snapshot.exports:
        return ArchiveDataFrames(
            messages=_empty_frame(MESSAGE_SCHEMA),
            users=_with_real_names(_empty_frame(USER_CACHE_SCHEMA), mappings),
            channels=_empty_frame(CHANNEL_SCHEMA),
        )

    message_index, cached_messages = _read_latest_cache(
        snapshot.caches["messages"], "messages"
    )
    user_index, cached_users = _read_latest_cache(
        snapshot.caches["users"], "users"
    )
    channel_index, cached_channels = _read_latest_cache(
        snapshot.caches["channels"], "channels"
    )

    records = _new_export_records(
        snapshot, min(message_index, user_index, channel_index)
    )
    fresh_messages = _frames_from_exports(
        _records_after(records, message_index)
    ).messages
    fresh_users = _frames_from_exports(
        _records_after(records, user_index)
    ).users
    fresh_channels = _frames_from_exports(
        _records_after(records, channel_index)
    ).channels
    cache_index = max(snapshot.exports)
    cached_frames = ArchiveDataFrames(
        messages=_merge_messages(cached_messages, fresh_messages),
        users=_merge_users(cached_users, fresh_users),
        channels=_merge_channels(cached_channels, fresh_channels),
    )
    _write_caches(archive_dir, cache_index, cached_frames, snapshot.caches)
    return ArchiveDataFrames(
        messages=cached_frames.messages,
        users=_with_real_names(cached_frames.users, mappings),
        channels=cached_frames.channels,
    )


if __name__ == "__main__":
    frames = load_dataframes()
    print("messages")
    print(frames.messages.schema)
    print("users")
    print(frames.users.schema)
    print("channels")
    print(frames.channels.schema)
