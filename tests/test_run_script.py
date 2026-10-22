"""scripts/run.sh explains a missing .env instead of exiting silently."""

import shutil
import subprocess
from pathlib import Path

RUN_SH = Path(__file__).resolve().parent.parent / "scripts" / "run.sh"


def test_run_sh_without_env_says_what_to_create(tmp_path: Path) -> None:
    # A copy in a repo-shaped folder with no .env; run.sh cds to its parent.
    (tmp_path / "scripts").mkdir()
    shutil.copy(RUN_SH, tmp_path / "scripts" / "run.sh")

    result = subprocess.run(
        ["bash", str(tmp_path / "scripts" / "run.sh")],
        cwd="/",
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 1
    assert "no .env" in result.stderr
    assert "cp .env.template .env" in result.stderr
    assert result.stdout == ""  # stopped before launching anything
