from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

from .urdf import UrdfInfo, parse_urdf


@dataclass(frozen=True)
class RobotAsset:
    """Resolved floating-base robot asset metadata.

    Attributes:
        name: Canonical asset name or direct URDF stem.
        urdf_path: Path to the original URDF asset.
        adam_urdf_path: Path to the Adam-compatible URDF used for dynamics.
        root_link: Floating-base/root link name parsed from the URDF.
        joint_names: Ordered movable joint names. Tensor joint coordinates
            with shape ``(..., n_joints)`` use exactly this order.
        default_contact_links: Link names used for default foot contact
            geometry.
        urdf: Parsed URDF metadata.
    """

    name: str
    urdf_path: Path
    adam_urdf_path: Path
    root_link: str
    joint_names: tuple[str, ...]
    default_contact_links: tuple[str, ...]
    urdf: UrdfInfo


_ASSET_VARIANTS = {
    "unitree_g1": "g1_29dof_rev_1_0.urdf",
    "g1": "g1_29dof_rev_1_0.urdf",
    "g1_29dof_rev_1_0": "g1_29dof_rev_1_0.urdf",
    "g1_29dof_mode_11": "g1_29dof_mode_11.urdf",
    "g1_29dof_mode_12": "g1_29dof_mode_12.urdf",
    "g1_29dof_mode_13": "g1_29dof_mode_13.urdf",
    "g1_29dof_mode_14": "g1_29dof_mode_14.urdf",
    "g1_29dof_mode_15": "g1_29dof_mode_15.urdf",
    "g1_29dof_mode_16": "g1_29dof_mode_16.urdf",
}

_UNITREE_G1_CONTACT_LINKS = ("left_ankle_roll_link", "right_ankle_roll_link")


def available_assets() -> tuple[str, ...]:
    """Return the registered built-in asset names.

    Args:
        None.

    Returns:
        Sorted tuple of asset aliases accepted by :func:`load_asset`.
    """
    return tuple(sorted(_ASSET_VARIANTS))


@lru_cache(maxsize=None)
def load_asset(asset_name: str = "unitree_g1") -> RobotAsset:
    """Resolve a built-in asset alias or a direct URDF path.

    Args:
        asset_name: Built-in asset alias such as ``"unitree_g1"`` or a direct
            path to a URDF file.

    Returns:
        :class:`RobotAsset` with parsed root link, joint order, contact
        links, original URDF path, and Adam-compatible URDF path.

    Raises:
        KeyError: If ``asset_name`` is neither a known alias nor an existing
            path.
        ValueError: If the resolved URDF cannot be parsed.
    """
    candidate = Path(asset_name).expanduser()
    if candidate.exists():
        urdf_path = candidate.resolve()
        canonical_name = urdf_path.stem
    else:
        if asset_name not in _ASSET_VARIANTS:
            raise KeyError(
                f"Unknown asset {asset_name!r}. Available assets: {', '.join(available_assets())}"
            )
        canonical_name = asset_name
        urdf_path = (
            resources.files("focodyn")
            / "assets"
            / "robots"
            / "unitree_g1"
            / _ASSET_VARIANTS[asset_name]
        )
        urdf_path = Path(str(urdf_path))

    info = parse_urdf(urdf_path)
    contact_links = tuple(
        link for link in _UNITREE_G1_CONTACT_LINKS if any(c.link_name == link for c in info.collisions)
    )
    return RobotAsset(
        name=canonical_name,
        urdf_path=urdf_path,
        adam_urdf_path=_adam_compatible_path(urdf_path),
        root_link=info.root_link,
        joint_names=info.joint_names,
        default_contact_links=contact_links,
        urdf=info,
    )


def _adam_compatible_path(urdf_path: Path) -> Path:
    """Return the generated Adam-compatible URDF path when available.

    Args:
        urdf_path: Path to an original URDF file.

    Returns:
        Sibling ``*.adam.urdf`` path if it exists, otherwise ``urdf_path``.
    """
    candidate = urdf_path.with_name(f"{urdf_path.stem}.adam.urdf")
    return candidate if candidate.exists() else urdf_path
