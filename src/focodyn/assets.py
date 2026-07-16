from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

from .joint_conventions import JointOrder, joint_names_for_order
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
        joint_order: Convention used for ``joint_names``. ``"source"`` keeps
            the URDF order; ``"unitree_g1_29dof"`` matches shared joints to
            the Unitree 29-DoF convention.
        default_contact_links: Link names used for default foot contact
            geometry.
        urdf: Parsed URDF metadata.
        usd_path: Optional path to the USD source from which the URDF
            companions were generated.
    """

    name: str
    urdf_path: Path
    adam_urdf_path: Path
    root_link: str
    joint_names: tuple[str, ...]
    default_contact_links: tuple[str, ...]
    urdf: UrdfInfo
    usd_path: Path | None = None
    joint_order: JointOrder = "source"

    @property
    def source_joint_names(self) -> tuple[str, ...]:
        """Return movable joint names in source/companion URDF order."""
        return self.urdf.joint_names


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
    "g1_37dof_minimal": "g1_37dof_minimal.urdf",
}

_ASSET_USD_VARIANTS = {
    "g1_37dof_minimal": "g1_37dof_minimal.usd",
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
def load_asset(
    asset_name: str = "unitree_g1", *, joint_order: JointOrder = "source"
) -> RobotAsset:
    """Resolve a built-in asset alias or a direct URDF/USD path.

    Args:
        asset_name: Built-in asset alias such as ``"unitree_g1"`` or a direct
            path to a URDF file. USD paths require same-stem generated URDF
            companions.
        joint_order: ``"source"`` preserves the movable-joint order of the
            URDF (including a USD asset's companion URDF).
            ``"unitree_g1_29dof"`` reorders shared G1 joints to match the
            Unitree 29-DoF convention and appends source-only joints in their
            original relative order.

    Returns:
        :class:`RobotAsset` with parsed root link, joint order, contact
        links, source paths, and Adam-compatible URDF path.

    Raises:
        KeyError: If ``asset_name`` is neither a known alias nor an existing
            path.
        ValueError: If the resolved URDF cannot be parsed or the requested
            joint convention is invalid for the asset.
    """
    candidate = Path(asset_name).expanduser()
    usd_path: Path | None = None
    if candidate.exists():
        candidate = candidate.resolve()
        canonical_name = candidate.stem
        if candidate.suffix.lower() == ".usd":
            usd_path = candidate
            urdf_path = candidate.with_suffix(".urdf")
            if not urdf_path.exists():
                raise ValueError(
                    f"USD asset {candidate} requires a same-stem URDF companion at {urdf_path}. "
                    "Runtime USD conversion is not supported."
                )
        else:
            urdf_path = candidate
            sibling_usd = candidate.with_suffix(".usd")
            usd_path = sibling_usd if sibling_usd.exists() else None
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
        usd_filename = _ASSET_USD_VARIANTS.get(asset_name)
        if usd_filename is not None:
            usd_path = urdf_path.with_name(usd_filename)

    info = parse_urdf(urdf_path)
    joint_names = joint_names_for_order(info.joint_names, joint_order)
    contact_links = tuple(
        link
        for link in _UNITREE_G1_CONTACT_LINKS
        if any(c.link_name == link for c in info.collisions)
    )
    return RobotAsset(
        name=canonical_name,
        urdf_path=urdf_path,
        adam_urdf_path=_adam_compatible_path(urdf_path),
        root_link=info.root_link,
        joint_names=joint_names,
        default_contact_links=contact_links,
        urdf=info,
        usd_path=usd_path,
        joint_order=joint_order,
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
