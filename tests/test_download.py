from __future__ import annotations

import hashlib
import http.server
import io
import tarfile
import threading
from pathlib import Path

import pytest

from cwfc import download
from cwfc.download import (
    CHECKSUM_FILE,
    REQUIRED_FILES,
    download_data,
    main,
    parse_checksums,
    sha256_of,
    verify_checksums,
)

ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _digest(data: bytes) -> str:
    """Return the SHA-256 hex digest of some bytes."""
    return hashlib.sha256(data).hexdigest()


def _payload() -> dict[str, bytes]:
    """Return fake contents for each required data file."""
    return {name: f"content of {name}".encode() for name in REQUIRED_FILES}


def _sums_text(files: dict[str, bytes]) -> bytes:
    """Return a SHA256SUMS file body listing the given files."""
    return "".join(f"{_digest(data)}  {name}\n" for name, data in files.items()).encode()


def _make_archive(path: Path, entries: dict[str, bytes]) -> Path:
    """Write a .tar.gz containing the given name-to-content entries."""
    with tarfile.open(path, "w:gz") as archive:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


def _write_release(folder: Path) -> Path:
    """Build a small valid release archive with matching checksums."""
    files = _payload()
    return _make_archive(folder / "data-v1.tar.gz", {**files, CHECKSUM_FILE: _sums_text(files)})


def _write_valid_folder(folder: Path) -> Path:
    """Create a folder holding valid release files and return it."""
    folder.mkdir(parents=True, exist_ok=True)
    files = _payload()
    for name, data in files.items():
        (folder / name).write_bytes(data)
    (folder / CHECKSUM_FILE).write_bytes(_sums_text(files))
    return folder


# sha256_of


def test_sha256_of_matches_known_digests(tmp_path):
    """The digest of known contents matches the published SHA-256 values."""
    (tmp_path / "abc.txt").write_bytes(b"abc")
    (tmp_path / "empty.txt").write_bytes(b"")
    assert sha256_of(tmp_path / "abc.txt") == ABC_SHA256
    assert sha256_of(tmp_path / "empty.txt") == EMPTY_SHA256


def test_sha256_of_missing_file_raises(tmp_path):
    """A path that does not exist raises a clear FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="file not found"):
        sha256_of(tmp_path / "nope.bin")


def test_sha256_of_directory_raises(tmp_path):
    """A directory is not hashable and raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        sha256_of(tmp_path)


# parse_checksums


def test_parse_checksums_handles_text_and_binary_markers(tmp_path):
    """Lines with or without the binary marker are parsed and lowercased."""
    path = tmp_path / CHECKSUM_FILE
    upper = ABC_SHA256.upper()
    path.write_text(f"{upper}  one.csv\n{EMPTY_SHA256} *two.csv\n\n", encoding="utf-8")
    assert parse_checksums(path) == {"one.csv": ABC_SHA256, "two.csv": EMPTY_SHA256}


def test_parse_checksums_ignores_utf8_byte_order_mark(tmp_path):
    """A checksum file saved with a UTF-8 BOM (as PowerShell does) still parses."""
    path = tmp_path / CHECKSUM_FILE
    path.write_bytes(b"\xef\xbb\xbf" + f"{ABC_SHA256}  one.csv\n".encode())
    assert parse_checksums(path) == {"one.csv": ABC_SHA256}


@pytest.mark.parametrize("second_digest", [ABC_SHA256, EMPTY_SHA256])
def test_parse_checksums_rejects_duplicate_file_names(tmp_path, second_digest):
    """A file listed twice is rejected, whether or not the digests agree."""
    path = tmp_path / CHECKSUM_FILE
    path.write_text(f"{ABC_SHA256}  one.csv\n{second_digest}  one.csv\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2 lists one.csv more than once"):
        parse_checksums(path)


def test_parse_checksums_missing_file_raises(tmp_path):
    """A missing checksum file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="checksum file not found"):
        parse_checksums(tmp_path / CHECKSUM_FILE)


def test_parse_checksums_empty_file_gives_empty_mapping(tmp_path):
    """An empty checksum file parses to an empty mapping."""
    path = tmp_path / CHECKSUM_FILE
    path.write_text("", encoding="utf-8")
    assert parse_checksums(path) == {}


@pytest.mark.parametrize(
    "line",
    [
        "abc123  file.csv",  # digest too short
        f"{'g' * 64}  file.csv",  # not hexadecimal
        ABC_SHA256,  # no file name
        f"{ABC_SHA256}   ",  # blank file name
        "just some words",
    ],
)
def test_parse_checksums_rejects_malformed_lines(tmp_path, line):
    """Lines that are not '<sha256>  <file name>' raise ValueError."""
    path = tmp_path / CHECKSUM_FILE
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        parse_checksums(path)


@pytest.mark.parametrize("name", ["../escape.csv", "sub/file.csv", "/abs/file.csv"])
def test_parse_checksums_rejects_names_with_paths(tmp_path, name):
    """File names that contain a directory part are rejected."""
    path = tmp_path / CHECKSUM_FILE
    path.write_text(f"{ABC_SHA256}  {name}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="plain file name"):
        parse_checksums(path)


# verify_checksums


def test_verify_passes_for_valid_folder(tmp_path):
    """A folder with correct files and checksums has no problems."""
    assert verify_checksums(_write_valid_folder(tmp_path / "data")) == []


def test_verify_reports_missing_checksum_file(tmp_path):
    """An empty folder is reported as missing its checksum file."""
    assert verify_checksums(tmp_path) == [f"missing {CHECKSUM_FILE}"]


def test_verify_reports_missing_folder(tmp_path):
    """A folder that does not exist is reported, not raised."""
    assert verify_checksums(tmp_path / "does_not_exist") == [f"missing {CHECKSUM_FILE}"]


def test_verify_reports_malformed_checksum_file(tmp_path):
    """A malformed checksum file is reported as a problem."""
    folder = _write_valid_folder(tmp_path / "data")
    (folder / CHECKSUM_FILE).write_text("garbage\n", encoding="utf-8")
    problems = verify_checksums(folder)
    assert len(problems) == 1 and problems[0].startswith(f"malformed {CHECKSUM_FILE}")


def test_verify_reports_undecodable_checksum_file(tmp_path):
    """A checksum file that is not valid UTF-8 is reported as malformed."""
    folder = _write_valid_folder(tmp_path / "data")
    (folder / CHECKSUM_FILE).write_bytes(b"\xff\xfe\x00bad")
    problems = verify_checksums(folder)
    assert len(problems) == 1 and problems[0].startswith(f"malformed {CHECKSUM_FILE}")


def test_verify_reports_unreadable_data_file(tmp_path, monkeypatch):
    """A data file that cannot be read is reported instead of raising."""
    folder = _write_valid_folder(tmp_path / "data")

    def deny(path):
        """Pretend the file is locked."""
        raise PermissionError("access denied")

    monkeypatch.setattr(download, "sha256_of", deny)
    problems = verify_checksums(folder)
    assert problems == [f"cannot read {name}: access denied" for name in REQUIRED_FILES]


def test_verify_reports_unreadable_checksum_file(tmp_path, monkeypatch):
    """A checksum file that cannot be read is reported instead of raising."""
    folder = _write_valid_folder(tmp_path / "data")

    def deny(path):
        """Pretend the checksum file is locked."""
        raise PermissionError("access denied")

    monkeypatch.setattr(download, "parse_checksums", deny)
    assert verify_checksums(folder) == [f"cannot read {CHECKSUM_FILE}: access denied"]


def test_verify_reports_missing_required_file_once(tmp_path):
    """A missing required file is reported a single time."""
    folder = _write_valid_folder(tmp_path / "data")
    (folder / REQUIRED_FILES[0]).unlink()
    assert verify_checksums(folder) == [f"missing {REQUIRED_FILES[0]}"]


def test_verify_reports_all_missing_required_files(tmp_path):
    """Several missing required files are each reported."""
    folder = _write_valid_folder(tmp_path / "data")
    for name in REQUIRED_FILES:
        (folder / name).unlink()
    assert verify_checksums(folder) == [f"missing {name}" for name in REQUIRED_FILES]


def test_verify_reports_required_file_not_listed(tmp_path):
    """A required file absent from the checksum list is reported."""
    folder = _write_valid_folder(tmp_path / "data")
    files = {k: v for k, v in _payload().items() if k != REQUIRED_FILES[1]}
    (folder / CHECKSUM_FILE).write_bytes(_sums_text(files))
    assert verify_checksums(folder) == [f"{REQUIRED_FILES[1]} is not listed in {CHECKSUM_FILE}"]


def test_verify_reports_empty_checksum_file(tmp_path):
    """An empty checksum file leaves every required file unlisted."""
    folder = _write_valid_folder(tmp_path / "data")
    (folder / CHECKSUM_FILE).write_text("", encoding="utf-8")
    assert verify_checksums(folder) == [f"{name} is not listed in {CHECKSUM_FILE}" for name in REQUIRED_FILES]


def test_verify_reports_listed_extra_file_that_is_missing(tmp_path):
    """A non-required file listed in the checksums but absent is reported."""
    folder = _write_valid_folder(tmp_path / "data")
    with (folder / CHECKSUM_FILE).open("ab") as handle:
        handle.write(f"{ABC_SHA256}  extra.bin\n".encode())
    assert verify_checksums(folder) == ["missing extra.bin"]


def test_verify_accepts_listed_extra_file_that_exists(tmp_path):
    """A non-required file listed with the right digest is accepted."""
    folder = _write_valid_folder(tmp_path / "data")
    (folder / "extra.bin").write_bytes(b"abc")
    with (folder / CHECKSUM_FILE).open("ab") as handle:
        handle.write(f"{ABC_SHA256}  extra.bin\n".encode())
    assert verify_checksums(folder) == []


def test_verify_reports_checksum_mismatch(tmp_path):
    """A changed file is reported as a checksum mismatch."""
    folder = _write_valid_folder(tmp_path / "data")
    (folder / REQUIRED_FILES[0]).write_bytes(b"corrupted")
    assert verify_checksums(folder) == [f"checksum mismatch for {REQUIRED_FILES[0]}"]


def test_verify_treats_directory_named_like_a_file_as_missing(tmp_path):
    """A directory in place of a required file counts as missing."""
    folder = _write_valid_folder(tmp_path / "data")
    (folder / REQUIRED_FILES[2]).unlink()
    (folder / REQUIRED_FILES[2]).mkdir()
    assert verify_checksums(folder) == [f"missing {REQUIRED_FILES[2]}"]


# download_data


def test_download_extracts_and_verifies_local_archive(tmp_path):
    """A valid archive is extracted, verified and leaves no temporary files."""
    archive = _write_release(tmp_path)
    dest = tmp_path / "data"
    result = download_data(dest, url=archive.as_uri())
    assert result == dest
    assert verify_checksums(dest) == []
    assert sorted(p.name for p in dest.iterdir()) == sorted([*REQUIRED_FILES, CHECKSUM_FILE])


def test_download_accepts_string_destination(tmp_path):
    """The destination may be given as a plain string."""
    archive = _write_release(tmp_path)
    assert download_data(str(tmp_path / "data"), url=archive.as_uri()) == tmp_path / "data"


def test_download_creates_nested_destination(tmp_path):
    """Missing parent folders of the destination are created."""
    archive = _write_release(tmp_path)
    dest = tmp_path / "a" / "b" / "data"
    download_data(dest, url=archive.as_uri())
    assert verify_checksums(dest) == []


def test_download_is_skipped_when_files_are_verified(tmp_path):
    """No download is attempted when verified files already exist."""
    dest = _write_valid_folder(tmp_path / "data")
    # A missing archive would fail if a download were attempted.
    assert download_data(dest, url=(tmp_path / "missing.tar.gz").as_uri()) == dest


def test_force_downloads_even_when_verified(tmp_path):
    """With force, a download is attempted even if the files are verified."""
    dest = _write_valid_folder(tmp_path / "data")
    with pytest.raises(RuntimeError, match="could not download"):
        download_data(dest, url=(tmp_path / "missing.tar.gz").as_uri(), force=True)


def test_corrupted_file_is_detected_and_redownloaded(tmp_path):
    """A corrupted file is flagged and fixed by downloading again."""
    archive = _write_release(tmp_path)
    dest = _write_valid_folder(tmp_path / "data")
    (dest / REQUIRED_FILES[0]).write_bytes(b"corrupted")
    assert verify_checksums(dest) == [f"checksum mismatch for {REQUIRED_FILES[0]}"]
    download_data(dest, url=archive.as_uri())
    assert verify_checksums(dest) == []


def test_missing_file_is_redownloaded(tmp_path):
    """A deleted data file is restored by downloading again."""
    archive = _write_release(tmp_path)
    dest = _write_valid_folder(tmp_path / "data")
    (dest / REQUIRED_FILES[1]).unlink()
    download_data(dest, url=archive.as_uri())
    assert (dest / REQUIRED_FILES[1]).is_file() and verify_checksums(dest) == []


def test_missing_archive_raises_clear_error_and_leaves_no_temp_files(tmp_path):
    """A URL pointing at a missing file raises RuntimeError and cleans up."""
    dest = tmp_path / "data"
    with pytest.raises(RuntimeError, match="could not download"):
        download_data(dest, url=(tmp_path / "missing.tar.gz").as_uri())
    assert list(dest.iterdir()) == []


def test_dropped_connection_raises_clear_error(tmp_path):
    """A chunked response cut off mid-body raises RuntimeError, not IncompleteRead."""

    class CutOffHandler(http.server.BaseHTTPRequestHandler):
        """Announce a chunk of 100 bytes, send 5, and hang up."""

        def do_GET(self):
            """Answer every GET with a truncated chunked body."""
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self.wfile.write(b"64\r\nshort")
            self.wfile.flush()
            self.close_connection = True

        def log_message(self, *args):
            """Keep the test output quiet."""

    server = http.server.HTTPServer(("127.0.0.1", 0), CutOffHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(RuntimeError, match="could not download"):
            download_data(tmp_path / "data", url=f"http://127.0.0.1:{server.server_port}/data.tar.gz")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_malformed_url_raises_clear_error(tmp_path):
    """A string that is not a URL raises RuntimeError, not ValueError."""
    with pytest.raises(RuntimeError, match="could not download"):
        download_data(tmp_path / "data", url="not a url")


def test_non_archive_download_raises_clear_error(tmp_path):
    """A downloaded file that is not a .tar.gz raises RuntimeError."""
    bogus = tmp_path / "bogus.tar.gz"
    bogus.write_bytes(b"this is not an archive")
    with pytest.raises(RuntimeError, match="not a valid .tar.gz"):
        download_data(tmp_path / "data", url=bogus.as_uri())


def test_truncated_archive_raises_clear_error(tmp_path):
    """An archive cut off mid-file raises RuntimeError."""
    good = _write_release(tmp_path).read_bytes()
    truncated = tmp_path / "truncated.tar.gz"
    truncated.write_bytes(good[: len(good) // 2])
    with pytest.raises(RuntimeError, match="not a valid .tar.gz"):
        download_data(tmp_path / "data", url=truncated.as_uri())


def test_archive_missing_required_files_raises(tmp_path):
    """An archive without the expected files fails verification."""
    archive = _make_archive(tmp_path / "partial.tar.gz", {"other.txt": b"x"})
    with pytest.raises(RuntimeError, match="failed verification"):
        download_data(tmp_path / "data", url=archive.as_uri())


def test_bad_checksum_in_archive_raises(tmp_path):
    """An archive whose checksums do not match its files raises an error."""
    entries = {name: b"x" for name in REQUIRED_FILES}
    entries[CHECKSUM_FILE] = "".join(f"{'0' * 64}  {name}\n" for name in REQUIRED_FILES).encode()
    archive = _make_archive(tmp_path / "bad.tar.gz", entries)
    with pytest.raises(RuntimeError, match="failed verification"):
        download_data(tmp_path / "data", url=archive.as_uri())


@pytest.mark.parametrize("name", ["../escape.txt", "/tmp/absolute_escape.txt"])
def test_archive_with_escaping_path_is_rejected(tmp_path, name):
    """Archive entries that escape the target folder are refused."""
    archive = _make_archive(tmp_path / "evil.tar.gz", {name: b"x"})
    with pytest.raises(RuntimeError, match="unsafe path"):
        download_data(tmp_path / "data", url=archive.as_uri())
    assert not (tmp_path / "escape.txt").exists()


def test_archive_with_symlink_is_rejected(tmp_path):
    """Symbolic links inside the archive are refused."""
    archive_path = tmp_path / "link.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "elsewhere"
        archive.addfile(info)
    with pytest.raises(RuntimeError, match="unsupported entry"):
        download_data(tmp_path / "data", url=archive_path.as_uri())


def test_destination_that_is_a_file_raises(tmp_path):
    """A destination path that is an existing file raises NotADirectoryError."""
    target = tmp_path / "data"
    target.write_text("i am a file", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        download_data(target, url=(tmp_path / "x.tar.gz").as_uri())


# default_data_dir


def test_default_data_dir_is_challenge_data_at_repo_root():
    """The default data folder is challenge_data next to pyproject.toml."""
    path = download.default_data_dir()
    assert path.name == "challenge_data"
    assert path.parent.joinpath("pyproject.toml").is_file()


def test_default_data_dir_falls_back_to_working_directory(tmp_path, monkeypatch):
    """Outside a source checkout the default is ./challenge_data."""
    fake_module = tmp_path / "site" / "pkg" / "cwfc" / "download.py"
    fake_module.parent.mkdir(parents=True)
    fake_module.write_text("", encoding="utf-8")
    monkeypatch.setattr(download, "__file__", str(fake_module))
    monkeypatch.chdir(tmp_path)
    assert download.default_data_dir() == tmp_path / "challenge_data"


# command line


def test_main_downloads_then_reports_already_present(tmp_path, capsys):
    """The CLI downloads once, then reports the data is already there."""
    archive = _write_release(tmp_path)
    args = ["--dest", str(tmp_path / "data"), "--url", archive.as_uri()]
    assert main(args) == 0
    assert "downloaded and verified" in capsys.readouterr().out
    assert main(args) == 0
    assert "already present and verified" in capsys.readouterr().out


def test_main_force_downloads_again(tmp_path, capsys):
    """The --force flag downloads even when verified files exist."""
    archive = _write_release(tmp_path)
    args = ["--dest", str(tmp_path / "data"), "--url", archive.as_uri()]
    main(args)
    capsys.readouterr()
    assert main([*args, "--force"]) == 0
    assert "downloaded and verified" in capsys.readouterr().out


def test_main_returns_error_code_on_failure(tmp_path, capsys):
    """The CLI prints a clean error and returns 1 when the download fails."""
    args = ["--dest", str(tmp_path / "data"), "--url", (tmp_path / "missing.tar.gz").as_uri()]
    assert main(args) == 1
    assert "error: could not download" in capsys.readouterr().err
