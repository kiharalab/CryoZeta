import os
from pathlib import Path


def get_repo_root() -> Path:
    """Return the CryoZeta repository root for source/editable installs."""
    return Path(__file__).resolve().parent.parent.parent.parent


def get_assets_dir() -> Path:
    """Return the preferred assets directory for runtime data files."""
    assets_dir = os.getenv("CRYOZETA_ASSETS_DIR")
    if assets_dir:
        return Path(assets_dir).expanduser().resolve()

    project_root = os.getenv("PIXI_PROJECT_ROOT")
    if project_root:
        return (Path(project_root).expanduser().resolve() / "assets").resolve()

    return get_repo_root() / "assets"


def resolve_asset_path(path: str | Path) -> Path:
    """Resolve a CryoZeta asset path independently of the current directory."""
    asset_path = Path(path).expanduser()
    if asset_path.is_absolute():
        return asset_path.resolve()

    parts = asset_path.parts
    if parts and parts[0] == "assets":
        asset_path = Path(*parts[1:])

    return (get_assets_dir() / asset_path).resolve()
