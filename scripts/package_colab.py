"""Package current source for Colab, excluding results, environments and notebook outputs."""

import argparse
import json
from pathlib import Path
import zipfile


def package(root, output, sweet=False, tuning=False):
    root, output = Path(root), Path(output)
    if sweet and tuning:
        raise ValueError('Choose one Colab workflow')
    prefix = 'CodeGuardLLM-TUNING' if tuning else 'CodeGuardLLM-SWEET' if sweet else 'CodeGuardLLM'
    files = [root / 'pyproject.toml', root / 'README.md']
    for folder in ('src', 'scripts', 'configs', 'sandbox', 'data', 'docs', 'tests', 'notebooks'):
        files.extend(p for p in (root / folder).rglob('*') if p.is_file()
                     and p.suffix in ('.py', '.yaml', '.jsonl', '.md', '.ipynb')
                     and '__pycache__' not in p.parts and '.ipynb_checkpoints' not in p.parts
                     and not any(part.endswith('.egg-info') for part in p.parts))
    files.append(root / 'sandbox/Dockerfile')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            content = path.read_bytes()
            if path.suffix == '.ipynb':
                notebook = json.loads(content)
                for cell in notebook['cells']:
                    if cell['cell_type'] == 'code':
                        cell.update(outputs=[], execution_count=None)
                content = (json.dumps(notebook, indent=1, ensure_ascii=False) + '\n').encode('utf-8')
            archive.writestr(prefix + '/' + path.relative_to(root).as_posix(), content)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    workflow = parser.add_mutually_exclusive_group()
    workflow.add_argument('--sweet', action='store_true')
    workflow.add_argument('--tuning', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    name = 'codeguard-tuning-colab.zip' if args.tuning else 'codeguard-sweet-colab.zip' if args.sweet else 'codeguard-colab.zip'
    print(package(root, root / '.tmp' / name, sweet=args.sweet, tuning=args.tuning))
