"""Allow `python -m wattwise` to print the insights report."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main(["insights"]))
