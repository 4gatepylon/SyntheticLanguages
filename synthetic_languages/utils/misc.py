from __future__ import annotations

from pathlib import Path


def repo_path_to_abs_path(path: str) -> Path:
    """
    Convert a path relative to the repository root to an absolute path.

    Args:
        path: A path relative to the repository root.

    Returns:
        The absolute path.
    """

    # NOTE that __file__ in python is the __file__ form which this was
    # invoked/imported, so if we are running this from a certain location, it may not
    # be the same as THIS file's location.
    repo_abs_path = Path(__file__).parent.parent.parent.absolute()
    return repo_abs_path / path
