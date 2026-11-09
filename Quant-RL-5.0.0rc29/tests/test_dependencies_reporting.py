"""Installation metadata and vector report regression."""
from pathlib import Path
import ast
import tomllib

from quant_rl.reporting.economic_report import _render_pdf

ROOT = Path(__file__).resolve().parents[1]


def test_requirements_match_runtime_project_dependencies():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    requirements = [line.strip() for line in (ROOT / 'requirements.txt').read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith('#')]
    assert requirements == ['-e .']
    assert project['project']['dependencies']
    assert any('MetaTrader5' in r and 'win32' in r for r in project['project']['dependencies'])
    assert not any('matplotlib' in r.lower() or 'scikit-learn' in r.lower()
                   for r in project['project']['dependencies'])


def test_pdf_renderer_does_not_load_matplotlib():
    import inspect
    source = inspect.getsource(_render_pdf)
    tree = ast.parse(source)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    assert 'matplotlib' not in imports and 'matplotlib.pyplot' not in imports
    assert 'reportlab.graphics.shapes' in imports


def test_dev_requirements_reference_runtime():
    lines = (ROOT / 'requirements-dev.txt').read_text().splitlines()
    assert '-e .[dev]' in lines
    assert any(r.startswith('pytest') for r in __import__('tomllib').loads((ROOT/'pyproject.toml').read_text())['project']['optional-dependencies']['dev'])
