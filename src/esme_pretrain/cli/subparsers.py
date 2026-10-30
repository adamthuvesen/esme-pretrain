from __future__ import annotations

import argparse

# argparse exposes no public name for the object that add_subparsers() returns.
type Subparsers = argparse._SubParsersAction[argparse.ArgumentParser]  # pyright: ignore[reportPrivateUsage]
