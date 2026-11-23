"""Reproducibility and output paths shared across experiment modules."""

from pathlib import Path

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = (PROJECT_ROOT if (PROJECT_ROOT / "pyproject.toml").is_file() else Path.cwd()) / "outputs"
FIGURES_DIR = OUTPUT_ROOT / "figures"
TABLES_DIR = OUTPUT_ROOT / "tables"
LOGS_DIR = OUTPUT_ROOT / "logs"


def ensure_output_dirs(output_root=OUTPUT_ROOT) -> None:
    """Create output directories relative to the project, regardless of cwd."""
    for name in ("figures", "tables", "logs"):
        (Path(output_root) / name).mkdir(parents=True, exist_ok=True)

