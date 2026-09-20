"""Load DiscordChatExporter archives from ``qter-channels`` into Polars.

Run this module directly to print the three dataframe schemas, or import
``load_dataframes`` from another script.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
CHANNEL_SCHEMA: dict[str, pl.DataType] = {
    "channel_name": pl.String,
    "channel_id": pl.String,
    "category_name": pl.String,
    "category_id": pl.String,
    "guild_name": pl.String,
    "guild_id": pl.String,
}


@dataclass(frozen=True)
class ArchiveDataFrames:
    messages: pl.DataFrame
    users: pl.DataFrame
    channels: pl.DataFrame


def _read_real_name_mappings(path: Path) -> dict[str, str]:
    """Read ``username -> Full Real Name`` mappings, ignoring comments."""
    if not path.exists():
        return {}

    mappings: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "->" not in line:
            raise ValueError(
                f"{path}:{line_number}: expected 'username -> Full Real Name'"
            )
        username, full_name = (part.strip() for part in line.split("->", maxsplit=1))
        if not username or not full_name:
            raise ValueError(
                f"{path}:{line_number}: username and full real name are both required"
            )
        mappings.setdefault(username.casefold(), full_name)
    return mappings


def _archive_files(archive_dir: Path) -> list[Path]:
    """Return JSON exports only; media directories are never traversed."""
    return sorted(
        path
        for path in archive_dir.rglob("*.json")
        if "media" not in path.relative_to(archive_dir).parts
    )


def _parse_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _string(value: Any) -> str | None:
    return None if value is None else str(value)


def load_dataframes(
    archive_dir: Path = ARCHIVE_DIR,
    name_mappings_file: Path = NAME_MAPPINGS_FILE,
) -> ArchiveDataFrames:
    """Load all Discord JSON exports below ``archive_dir`` into three frames.

    ``archive_dir`` may contain exports from any number of Discord servers.
    The loader only reads ``*.json`` files and explicitly excludes every path
    under a directory named ``media``.
    """
    mappings = _read_real_name_mappings(name_mappings_file)
    message_rows: list[dict[str, Any]] = []
    user_rows: list[dict[str, str | None]] = []
    channel_rows: list[dict[str, str | None]] = []

    files = _archive_files(archive_dir)

    for export_path in files:
        with export_path.open(encoding="utf-8") as export_file:
            export = json.load(export_file)

        guild = export.get("guild") or {}
        channel = export.get("channel") or {}
        channel_rows.append(
            {
                "channel_name": _string(channel.get("name")),
                "channel_id": _string(channel.get("id")),
                "category_name": _string(channel.get("category")),
                "category_id": _string(channel.get("categoryId")),
                "guild_name": _string(guild.get("name")),
                "guild_id": _string(guild.get("id")),
            }
        )

        for message in export.get("messages", []):
            author = message.get("author") or {}
            author_id = _string(author.get("id"))
            message_rows.append(
                {
                    "author_id": author_id,
                    "content": _string(message.get("content")),
                    "timestamp": _parse_timestamp(message.get("timestamp")),
                    "message_id": _string(message.get("id")),
                    "guild": _string(guild.get("name")),
                    "category": _string(channel.get("category")),
                    "channel": _string(channel.get("id")),
                }
            )
            if author_id is not None:
                username = _string(author.get("name"))
                user_rows.append(
                    {
                        "user_id": author_id,
                        "username": username,
                        "nickname": _string(author.get("nickname")),
                        "full_real_name": (
                            mappings.get(username.casefold()) if username is not None else None
                        ),
                    }
                )

    messages = pl.DataFrame(message_rows, schema=MESSAGE_SCHEMA)
    users_source = pl.DataFrame(
        user_rows,
        schema={
            "user_id": pl.String,
            "username": pl.String,
            "nickname": pl.String,
            "full_real_name": pl.String,
        }
    )
    channels_source = pl.DataFrame(channel_rows, schema=CHANNEL_SCHEMA)

    # Aggregate identities lazily so a user seen under renamed Discord handles
    # has one record. The first matched username mapping supplies the real name.
    users = (
        users_source.lazy()
        .group_by("user_id", maintain_order=True)
        .agg(
            pl.col("username").drop_nulls().unique(maintain_order=True).alias("usernames"),
            pl.col("nickname").drop_nulls().unique(maintain_order=True).alias("nicknames"),
            pl.col("full_real_name").drop_nulls().first().alias("full_real_name"),
        )
        .select(*USER_SCHEMA)
        .collect()
    )
    channels = (
        channels_source.lazy()
        .unique(subset=["channel_id", "guild_id"], maintain_order=True)
        .select(*CHANNEL_SCHEMA)
        .collect()
    )

    return ArchiveDataFrames(messages=messages, users=users, channels=channels)


if __name__ == "__main__":
    frames = load_dataframes()
    print("messages")
    print(frames.messages.schema)
    print("users")
    print(frames.users.schema)
    print("channels")
    print(frames.channels.schema)
