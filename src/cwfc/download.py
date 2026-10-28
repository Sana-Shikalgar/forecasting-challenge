"""Download and verify the fixed ``data-v1`` challenge release.

Uses only the standard library. Run ``python -m cwfc.download`` or call
``download_data()``. Files land in ``challenge_data/`` at the repository root,
which is where the rest of the package reads them from.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

RELEASE_URL = "https://github.com/atomityhq/forecasting-challenge/releases/download/data-v1/data-v1.tar.gz"
CHECKSUM_FILE = "SHA256SUMS"
REQUIRED_FILES = ("vm_cpu_readings.csv.gz", "vm_metadata.csv.gz", "DATA_CARD.md")
_CHUNK = 1024 * 1024
_TIMEOUT_SECONDS = 60
_HEX = frozenset("0123456789abcdefABCDEF")


def default_data_dir() -> Path:
    """Return ``challenge_data/`` at the repository root.

    The path is built from the package location, so it does not depend on the
    current working directory. If the package is not running from a source
    checkout, fall back to ``./challenge_data``.
    """
    root = Path(__file__).resolve().parents[2]
    if (root / "pyproject.toml").is_file():
        return root / "challenge_data"
    return Path.cwd() / "challenge_data"


def sha256_of(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks.

    Raises ``FileNotFoundError`` if ``path`` is not an existing file.
    """
    if not path.is_file():
        raise FileNotFoundError(f"file not found: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_checksums(path: Path) -> dict[str, str]:
    """Parse a ``sha256sum`` file into ``{filename: hex digest}``.

    A leading UTF-8 byte order mark is ignored. Raises ``FileNotFoundError`` if the
    file is missing and ``ValueError`` if a line is not
    ``<64 hex characters>  <plain file name>`` or a file name appears twice.
    """
    if not path.is_file():
        raise FileNotFoundError(f"checksum file not found: {path}")
    out: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        digest, _, name = line.partition(" ")
        name = name.strip().lstrip("*")
        if len(digest) != 64 or not set(digest) <= _HEX or not name:
            raise ValueError(f"line {number} is not '<sha256>  <file name>'")
        if Path(name).name != name:
            raise ValueError(f"line {number} has a path instead of a plain file name: {name}")
        if name in out:
            raise ValueError(f"line {number} lists {name} more than once")
        out[name] = digest.lower()
    return out


def verify_checksums(dest: Path) -> list[str]:
    """Return a list of problems found in ``dest``; empty means verified.

    A missing folder, missing or unreadable files, a malformed checksum file and
    mismatching digests are all reported as problems instead of raising.
    """
    try:
        expected = parse_checksums(dest / CHECKSUM_FILE)
    except FileNotFoundError:
        return [f"missing {CHECKSUM_FILE}"]
    except ValueError as exc:
        return [f"malformed {CHECKSUM_FILE}: {exc}"]
    except OSError as exc:
        return [f"cannot read {CHECKSUM_FILE}: {exc}"]
    problems = [f"missing {name}" for name in REQUIRED_FILES if not (dest / name).is_file()]
    problems += [f"{name} is not listed in {CHECKSUM_FILE}" for name in REQUIRED_FILES if name not in expected]
    for name, digest in expected.items():
        path = dest / name
        if path.is_file():
            try:
                matches = sha256_of(path) == digest
            except OSError as exc:  # locked or permission denied
                problems.append(f"cannot read {name}: {exc}")
            else:
                if not matches:
                    problems.append(f"checksum mismatch for {name}")
        elif name not in REQUIRED_FILES:
            problems.append(f"missing {name}")
    return problems


def _safe_extract(archive: tarfile.TarFile, dest: Path) -> None:
    """Extract regular files only, refusing paths that escape ``dest``."""
    root = dest.resolve()
    for member in archive.getmembers():
        target = (root / member.name).resolve()
        if root != target and root not in target.parents:
            raise RuntimeError(f"unsafe path in archive: {member.name}")
        if not member.isfile():
            raise RuntimeError(f"unsupported entry in archive: {member.name}")
    archive.extractall(root)


def _fetch(url: str, target: Path) -> None:
    """Stream the file at ``url`` into ``target``.

    Raises ``RuntimeError`` if the URL is invalid, unreachable or the file is missing.
    """
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_SECONDS) as response, target.open("wb") as handle:
            shutil.copyfileobj(response, handle, _CHUNK)
    except (OSError, ValueError, http.client.HTTPException) as exc:  # URLError, HTTPError, timeout, malformed URL, dropped connection
        raise RuntimeError(f"could not download {url}: {exc}") from exc


def download_data(dest: str | Path | None = None, url: str = RELEASE_URL, force: bool = False) -> Path:
    """Download ``data-v1`` into ``dest`` and verify it.

    Does nothing when ``dest`` already holds verified files, unless ``force``.
    Returns the data directory. Raises ``NotADirectoryError`` if ``dest`` is a file,
    and ``RuntimeError`` if the download fails, the archive is invalid or unsafe,
    or the extracted files fail verification.
    """
    dest = Path(dest) if dest is not None else default_data_dir()
    if dest.exists() and not dest.is_dir():
        raise NotADirectoryError(f"{dest} exists and is not a directory")
    if not force and not verify_checksums(dest):
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest) as tmp:
        archive_path = Path(tmp) / "data-v1.tar.gz"
        _fetch(url, archive_path)
        try:
            with tarfile.open(archive_path, "r:gz") as archive:
                _safe_extract(archive, dest)
        except (tarfile.TarError, EOFError) as exc:  # not a gzip/tar file, or truncated
            raise RuntimeError(f"downloaded file is not a valid .tar.gz archive: {exc}") from exc
    problems = verify_checksums(dest)
    if problems:
        raise RuntimeError("downloaded data failed verification: " + "; ".join(problems))
    return dest


def main(argv: list[str] | None = None) -> int:
    """Run the download command line tool and return its exit code."""
    parser = argparse.ArgumentParser(description="Download and verify the data-v1 challenge release.")
    parser.add_argument("--dest", type=Path, default=None, help="target directory (default: challenge_data/ at the repo root)")
    parser.add_argument("--url", default=RELEASE_URL, help="release archive URL")
    parser.add_argument("--force", action="store_true", help="download again even if verified files exist")
    args = parser.parse_args(argv)
    target = args.dest if args.dest is not None else default_data_dir()
    already_there = not args.force and not verify_checksums(target)
    try:
        path = download_data(args.dest, url=args.url, force=args.force)
    except Exception as exc:  # report any failure as a clean message and exit code
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{'already present and verified' if already_there else 'downloaded and verified'}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
