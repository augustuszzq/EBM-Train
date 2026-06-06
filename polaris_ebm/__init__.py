"""Compatibility package for source-tree tests.

The original project is imported as ``polaris_ebm``.  This clean collaboration
snapshot keeps the source directories at repository root, so this package adds
the repository root to the package search path.  That lets imports such as
``polaris_ebm.scripts.current.ebm_train_sync_mode_a`` resolve without requiring
the checkout directory itself to be named ``polaris_ebm``.
"""

from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
__path__.append(str(_repo_root))

