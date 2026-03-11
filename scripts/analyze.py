#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from dissectlm.inspector import inspect_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deep model inspection")
    parser.add_argument("model")
    args = parser.parse_args()
    print(json.dumps(inspect_model(args.model), indent=2))


if __name__ == "__main__":
    main()
