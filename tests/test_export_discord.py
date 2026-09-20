import json
import subprocess
import tarfile
from pathlib import Path

import polars as pl
import pytest

import export_discord
from messages_dataframe import (
    ArchiveDataFrames,
    CHANNEL_SCHEMA,
    MESSAGE_SCHEMA,
    USER_SCHEMA,
)


def _frames() -> ArchiveDataFrames:
    return ArchiveDataFrames(
        messages=pl.DataFrame(
            {
                "author_id": ["author"],
                "content": ["cached"],
                "timestamp": ["2025-01-01T00:00:00+00:00"],
                "message_id": ["100"],
                "guild": ["Guild"],
                "category": ["Category"],
                "channel": ["1"],
            },
            schema=MESSAGE_SCHEMA,
        ),
        users=pl.DataFrame(schema=USER_SCHEMA),
        channels=pl.DataFrame(schema=CHANNEL_SCHEMA),
    )


@pytest.fixture
def export_path(tmp_path: Path) -> Path:
    (tmp_path / "media").mkdir()
    return tmp_path


def _fake_run_factory(
    failing_channel: str | None = None, forbidden_channel: str | None = None
):
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        assert kwargs["env"]["DISCORD_TOKEN"] == "secret"  # type: ignore[index]
        if command[1] == "channels":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="1 | general\n * 2 | Thread / archived | Archived\n",
                stderr="",
            )

        channel_id = command[command.index("--channel") + 1]
        output_directory = Path(command[command.index("--output") + 1])
        output_directory.mkdir(exist_ok=True)
        (output_directory / f"{channel_id}.json").write_text(
            json.dumps({"channel": channel_id}), encoding="utf-8"
        )
        return subprocess.CompletedProcess(
            command,
            1 if channel_id in {failing_channel, forbidden_channel} else 0,
            stdout="",
            stderr=(
                f"DiscordChatExporter.Core.Exceptions.DiscordChatExporterException: "
                f"Request to 'channels/{channel_id}' failed: forbidden."
                if channel_id == forbidden_channel
                else "export failed" if channel_id == failing_channel else ""
            ),
        )

    return fake_run, commands


def test_export_guild_publishes_json_tar_and_status(
    export_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_run, commands = _fake_run_factory()
    monkeypatch.setattr(export_discord, "load_dataframes", lambda _: _frames())
    monkeypatch.setattr(export_discord.subprocess, "run", fake_run)
    monkeypatch.setenv("DISCORD_CHAT_EXPORTER_CLI", "fake-dce")

    archive_path = export_discord.export_guild(export_path, "guild-1", "secret", "test")

    assert archive_path == export_path / "0-test.tar.gz"
    with tarfile.open(archive_path) as archive:
        assert sorted(archive.getnames()) == ["1/1.json", "2/2.json"]
    assert commands[0] == ["fake-dce", "channels", "--guild", "guild-1", "--include-threads", "all"]
    first_export, second_export = commands[1:]
    assert "secret" not in first_export + second_export
    assert first_export[first_export.index("--after") + 1] == "100"
    assert "--after" not in second_export
    output = capsys.readouterr().out
    assert "Starting channel 1 (0/2 channels exported; 0 skipped for permissions)" in output
    assert "Finished channel 2 (2/2 channels exported; 0 skipped for permissions)" in output
    assert "Export complete: 2 channels exported, 0 skipped for permissions (2 channels listed)" in output


def test_export_guild_publishes_successes_then_raises_on_failure(
    export_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_run, commands = _fake_run_factory(failing_channel="2")
    monkeypatch.setattr(export_discord, "load_dataframes", lambda _: _frames())
    monkeypatch.setattr(export_discord.subprocess, "run", fake_run)

    with pytest.raises(export_discord.ExportError) as raised:
        export_discord.export_guild(export_path, "guild-1", "secret", "partial")

    assert raised.value.archive_path == export_path / "0-partial.tar.gz"
    assert raised.value.failed_channel_ids == ["2"]
    with tarfile.open(raised.value.archive_path) as archive:
        assert archive.getnames() == ["1/1.json"]
    assert len(commands) == 3
    assert (
        "Finished channel 2 with failure (1/2 channels exported; 0 skipped for permissions)"
        in capsys.readouterr().out
    )


def test_export_guild_does_not_publish_when_first_channel_fails(
    export_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_run, _ = _fake_run_factory(failing_channel="1")
    monkeypatch.setattr(export_discord, "load_dataframes", lambda _: _frames())
    monkeypatch.setattr(export_discord.subprocess, "run", fake_run)

    with pytest.raises(export_discord.ExportError) as raised:
        export_discord.export_guild(export_path, "guild-1", "secret", "failed")

    assert raised.value.archive_path is None
    assert list(export_path.glob("*.tar.gz")) == []


def test_export_guild_skips_forbidden_channels(
    export_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_run, _ = _fake_run_factory(forbidden_channel="2")
    monkeypatch.setattr(export_discord, "load_dataframes", lambda _: _frames())
    monkeypatch.setattr(export_discord.subprocess, "run", fake_run)

    archive_path = export_discord.export_guild(export_path, "guild-1", "secret", "allowed")

    with tarfile.open(archive_path) as archive:
        assert archive.getnames() == ["1/1.json"]
    output = capsys.readouterr().out
    assert "Skipping channel 2: permission denied (1/2 channels exported; 1 skipped for permissions)" in output
    assert "Export complete: 1 channels exported, 1 skipped for permissions (2 channels listed)" in output
