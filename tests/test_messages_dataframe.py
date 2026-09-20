import io
import json
import tarfile
from pathlib import Path

import polars as pl
import pytest

from messages_dataframe import load_dataframes


def _export(
    message_id: str = "message-1",
    content: str = "hello",
    username: str = "alice",
) -> dict[str, object]:
    return {
        "guild": {"id": "guild-1", "name": "Guild"},
        "channel": {
            "id": "channel-1",
            "name": "general",
            "category": "Chat",
            "categoryId": "category-1",
        },
        "messages": [
            {
                "id": message_id,
                "content": content,
                "timestamp": "2025-01-01T00:00:00+00:00",
                "author": {
                    "id": "user-1",
                    "name": username,
                    "nickname": username,
                },
            }
        ],
    }


@pytest.fixture
def archive_dir(tmp_path: Path) -> Path:
    (tmp_path / "media").mkdir()
    return tmp_path


def _write_tarball(
    archive_dir: Path, name: str, members: dict[str, object]
) -> None:
    with tarfile.open(archive_dir / name, "w:gz") as archive:
        for member_name, member in members.items():
            payload = (
                json.dumps(member).encode()
                if isinstance(member, dict)
                else member
            )
            info = tarfile.TarInfo(member_name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_empty_archive_returns_schema_correct_empty_frames(
    archive_dir: Path,
) -> None:
    frames = load_dataframes(archive_dir)

    assert frames.messages.is_empty()
    assert frames.users.is_empty()
    assert frames.channels.is_empty()
    assert list(archive_dir.glob("*.parquet")) == []


def test_incremental_cache_deduplicates_and_refreshes_real_names(
    archive_dir: Path,
) -> None:
    _write_tarball(
        archive_dir,
        "00-first.tar.gz",
        {"first.json": _export(content="old", username="old")},
    )
    _write_tarball(
        archive_dir,
        "1-update.tar.gz",
        {"update.json": _export(content="new", username="new")},
    )
    mappings = archive_dir / "names.txt"
    mappings.write_text("new -> New Name\n", encoding="utf-8")

    first = load_dataframes(archive_dir, mappings)

    assert first.messages.height == 1
    assert first.messages.item(0, "content") == "new"
    assert first.users.item(0, "full_real_name") == "New Name"
    assert {path.name for path in archive_dir.glob("*.parquet")} == {
        "1-messages.parquet",
        "1-users.parquet",
        "1-channels.parquet",
    }
    assert (
        "full_real_name"
        not in pl.read_parquet(archive_dir / "1-users.parquet").schema
    )

    mappings.write_text("old -> Old Name\n", encoding="utf-8")
    second = load_dataframes(archive_dir, mappings)

    assert second.users.item(0, "full_real_name") == "Old Name"


def test_stale_message_cache_is_caught_up_independently(
    archive_dir: Path,
) -> None:
    _write_tarball(
        archive_dir,
        "0-first.tar.gz",
        {"first.json": _export(message_id="one")},
    )
    _write_tarball(
        archive_dir,
        "1-second.tar.gz",
        {"second.json": _export(message_id="two")},
    )
    load_dataframes(archive_dir)

    pl.read_parquet(archive_dir / "1-messages.parquet").head(1).write_parquet(
        archive_dir / "0-messages.parquet"
    )
    (archive_dir / "1-messages.parquet").unlink()

    frames = load_dataframes(archive_dir)

    assert frames.messages.get_column("message_id").sort().to_list() == [
        "one",
        "two",
    ]
    assert (archive_dir / "1-messages.parquet").is_file()


def test_corrupt_cache_is_rebuilt_from_exports(archive_dir: Path) -> None:
    _write_tarball(archive_dir, "0-first.tar.gz", {"first.json": _export()})
    load_dataframes(archive_dir)
    (archive_dir / "0-messages.parquet").write_text(
        "not parquet", encoding="utf-8"
    )

    frames = load_dataframes(archive_dir)

    assert frames.messages.item(0, "message_id") == "message-1"
    assert pl.read_parquet(archive_dir / "0-messages.parquet").height == 1


@pytest.mark.parametrize(
    ("tarballs", "message"),
    [
        ({"1-later.tar.gz": {"later.json": _export()}}, "missing indices"),
        (
            {
                "0-first.tar.gz": {"first.json": _export()},
                "00-duplicate.tar.gz": {"duplicate.json": _export()},
            },
            "multiple exports use index 0",
        ),
        (
            {"0-first.tar.gz": {"not-json.txt": b"not json"}},
            "non-JSON tar member",
        ),
    ],
)
def test_invalid_archive_structure_fails(
    archive_dir: Path, tarballs: dict[str, dict[str, object]], message: str
) -> None:
    for name, members in tarballs.items():
        _write_tarball(archive_dir, name, members)

    with pytest.raises(ValueError, match=message):
        load_dataframes(archive_dir)


def test_missing_message_id_fails(archive_dir: Path) -> None:
    malformed = _export()
    malformed["messages"][0].pop("id")  # type: ignore[index]
    _write_tarball(archive_dir, "0-malformed.tar.gz", {"bad.json": malformed})

    with pytest.raises(ValueError, match="missing message id"):
        load_dataframes(archive_dir)


def test_unknown_files_are_ignored(archive_dir: Path) -> None:
    _write_tarball(archive_dir, "0.tar.gz", {"export.json": _export()})
    (archive_dir / "notes.txt").write_text("keep me", encoding="utf-8")

    load_dataframes(archive_dir)

    assert (archive_dir / "notes.txt").read_text(encoding="utf-8") == "keep me"
