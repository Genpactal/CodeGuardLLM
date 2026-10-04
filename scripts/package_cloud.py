"""Build an uploadable source archive, excluding environments, keys, and results."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "dist" / "codeguard-week1.zip"
    output.parent.mkdir(exist_ok=True)
    folders = ("src/codeguard", "configs", "data", "notebooks", "scripts", "sandbox", "tests", "docs")
    files = [root / name for name in ("README.md", "pyproject.toml", ".gitignore")]
    for folder in folders:
        files.extend(path for path in (root / folder).rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc")
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, "CodeGuardLLM/" + path.relative_to(root).as_posix())
    print(output)


if __name__ == "__main__":
    main()
