"""Incrementally export one Discord guild with DiscordChatExporter.

The command-line interface gets its token from ``--token`` or
``DISCORD_TOKEN``.  ``DISCORD_CHAT_EXPORTER_CLI`` can override the
``DiscordChatExporter.Cli`` executable used for child processes.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from uuid import uuid4

import polars as pl

from messages_dataframe import ArchiveDataFrames, load_dataframes


EXPORT_NAME = re.compile(r"^(?P<index>\d+)(?:-.*)?\.tar\.gz$")
CHANNEL_LINE = re.compile(r"^\s*(?:\*\s*)?(?P<channel_id>\d+)\s+\|\s")
FORBIDDEN_CHANNEL = re.compile(r"Request to 'channels/\d+' failed: forbidden\.", re.IGNORECASE)
FORUM_PARENT_CHANNEL = re.compile(
    r"is a forum and cannot be exported directly\.\s*You need to pull its threads",
    re.IGNORECASE,
)


class ExportError(RuntimeError):
    """A guild export that may have published a partial archive."""

    def __init__(self, archive_path: Path | None, failed_channel_ids: list[str]):
        self.archive_path = archive_path
        self.failed_channel_ids = failed_channel_ids
        location = f"; partial archive: {archive_path}" if archive_path else ""
        super().__init__(f"failed to export channels: {', '.join(failed_channel_ids)}{location}")


@dataclass(frozen=True)
class _ChannelFailure:
    channel_id: str


def _dce_environment(token: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment["DISCORD_TOKEN"] = token
    return environment


def _dce_executable() -> str:
    return os.environ.get("DISCORD_CHAT_EXPORTER_CLI", "DiscordChatExporter.Cli")


@contextmanager
def _export_lock(export_path: Path) -> Iterator[None]:
    """Serialize local guild exporters when ``fcntl`` is available."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows has no fcntl.
        yield
        return

    with (export_path / ".discord-export.lock").open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _channel_ids(executable: str, guild_id: str, environment: dict[str, str]) -> list[str]:
    result = subprocess.run(
        [executable, "channels", "--guild", guild_id, "--include-threads", "all"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    result.check_returncode()
    channel_ids: list[str] = []
    for line in result.stdout.splitlines():
        match = CHANNEL_LINE.match(line)
        if match:
            channel_ids.append(match["channel_id"])
        else:
            raise ValueError(f"DiscordChatExporter printed a line not parseable as a channel: `{line}`")
    if not channel_ids and result.stdout.strip():
        raise ValueError("DiscordChatExporter returned no recognizable channel IDs")
    return channel_ids


def _latest_message_id(messages: pl.DataFrame, channel_id: str) -> str | None:
    latest = (
        messages.lazy()
        .filter(pl.col("channel") == channel_id)
        .with_columns(pl.col("message_id").cast(pl.UInt64).alias("_message_id_number"))
        .sort(
            ["timestamp", "_message_id_number"],
            descending=[True, True],
            nulls_last=True,
        )
        .select("message_id")
        .head(1)
        .collect()
    )
    return None if latest.is_empty() else latest.item(0, "message_id")


def _validate_label(label: str) -> None:
    if any(character in label for character in ("/", "\\")) or any(
        ord(character) < 32 or ord(character) == 127 for character in label
    ):
        raise ValueError("label may not contain path separators or control characters")


def _next_export_index(export_path: Path) -> int:
    indices = [
        int(match["index"])
        for path in export_path.iterdir()
        if (match := EXPORT_NAME.fullmatch(path.name))
    ]
    return max(indices, default=-1) + 1


def _is_forbidden_channel_error(error_output: str) -> bool:
    return bool(FORBIDDEN_CHANNEL.search(error_output))


def _is_skippable_channel_error(error_output: str) -> bool:
    return _is_forbidden_channel_error(error_output) or bool(FORUM_PARENT_CHANNEL.search(error_output))


def _progress(exported: int, total: int, skipped: int) -> str:
    return f"({exported}/{total} channels exported; {skipped} skipped for permissions)"


def _tar_json_files(source_directory: Path, destination: Path) -> None:
    with tarfile.open(destination, "w:gz") as archive:
        for path in sorted(source_directory.rglob("*.json")):
            if not path.is_file():
                continue
            archive.add(path, arcname=path.relative_to(source_directory), recursive=False)


def _publish_archive(export_path: Path, source_directory: Path, label: str) -> Path:
    index = _next_export_index(export_path)
    final_path = export_path / f"{index}-{label}.tar.gz"
    temporary_path = export_path / f".{index}-{uuid4().hex}.tar.gz.tmp"
    _tar_json_files(source_directory, temporary_path)
    os.replace(temporary_path, final_path)
    return final_path


def export_guild(
    export_path: Path,
    guild_id: str,
    token: str,
    label: str | None = None,
) -> Path:
    """Export a guild incrementally and return its published archive path.

    On a channel failure, completed channel exports are published if any exist,
    then :class:`ExportError` is raised with the partial archive path.
    """
    if not token:
        raise ValueError("a Discord token is required")
    export_path = Path(export_path)
    chosen_label = label if label is not None else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    _validate_label(chosen_label)
    environment = _dce_environment(token)
    executable = _dce_executable()

    with _export_lock(export_path):
        frames: ArchiveDataFrames = load_dataframes(export_path)
        channel_ids = _channel_ids(executable, str(guild_id), environment)
        successful_channel_ids: list[str] = []
        skipped_channel_ids: list[str] = []
        failures: list[_ChannelFailure] = []

        with tempfile.TemporaryDirectory(dir=export_path, prefix=".discord-export-") as temporary_name:
            temporary_directory = Path(temporary_name)
            total = len(channel_ids)
            for channel_id in channel_ids:
                print(
                    f"Starting channel {channel_id} "
                    f"{_progress(len(successful_channel_ids), total, len(skipped_channel_ids))}"
                )
                channel_directory = temporary_directory / channel_id
                channel_directory.mkdir()
                command = [
                    executable,
                    "export",
                    "--channel",
                    channel_id,
                    "--format",
                    "Json",
                    "--output",
                    f"{channel_directory}{os.sep}",
                    "--media",
                    "--reuse-media",
                    "true",
                    "--media-dir",
                    str(export_path / "media"),
                ]
                if latest_message_id := _latest_message_id(frames.messages, channel_id):
                    print(f"Using latest message {latest_message_id}")
                    command.extend(("--after", latest_message_id))

                result = subprocess.run(
                    command,
                    check=False,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=environment,
                )
                if result.stderr:
                    sys.stderr.write(result.stderr)
                if result.returncode:
                    shutil.rmtree(channel_directory)
                    if _is_skippable_channel_error(result.stderr or ""):
                        skipped_channel_ids.append(channel_id)
                        reason = (
                            "forum parent channel"
                            if FORUM_PARENT_CHANNEL.search(result.stderr or "")
                            else "permission denied"
                        )
                        print(
                            f"Skipping channel {channel_id}: {reason} "
                            f"{_progress(len(successful_channel_ids), total, len(skipped_channel_ids))}"
                        )
                        continue
                    failures.append(_ChannelFailure(channel_id))
                    print(
                        f"Finished channel {channel_id} with failure "
                        f"{_progress(len(successful_channel_ids), total, len(skipped_channel_ids))}"
                    )
                    break

                successful_channel_ids.append(channel_id)
                print(
                    f"Finished channel {channel_id} "
                    f"{_progress(len(successful_channel_ids), total, len(skipped_channel_ids))}"
                )

            if failures and not successful_channel_ids:
                raise ExportError(None, [failure.channel_id for failure in failures])

            archive_path = _publish_archive(export_path, temporary_directory, chosen_label)

        if failures:
            raise ExportError(archive_path, [failure.channel_id for failure in failures])
        print(
            f"Export complete: {len(successful_channel_ids)} channels exported, "
            f"{len(skipped_channel_ids)} skipped for permissions ({total} channels listed)"
        )
        return archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_path", type=Path)
    parser.add_argument("guild_id")
    parser.add_argument("--label")
    parser.add_argument("--token", default=os.environ.get("DISCORD_TOKEN"))
    arguments = parser.parse_args()
    if not arguments.token:
        parser.error("provide --token or set DISCORD_TOKEN")

    try:
        archive_path = export_guild(
            arguments.export_path,
            arguments.guild_id,
            arguments.token,
            arguments.label,
        )
    except ExportError as error:
        print(error)
        raise SystemExit(1) from error
    print(f"Published {archive_path}")


if __name__ == "__main__":
    main()
