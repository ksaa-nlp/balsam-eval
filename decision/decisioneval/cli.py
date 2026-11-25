"""Operator commands: seeded adapter rows for the backend, a local REQUEST_CONFIG, and run audits."""

from __future__ import annotations

import argparse
import json

from decisioneval.audit import audit_run_file


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="decisioneval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Replay a stored run from its raw records")
    audit.add_argument("path", help="A run file written by the runner")

    adapters = subparsers.add_parser("adapters", help="Seeded adapter versions")
    adapters_commands = adapters.add_subparsers(dest="adapters_command", required=True)
    adapters_commands.add_parser("seed-sql", help="Print the backend migration rows, hashes included")
    config = adapters_commands.add_parser("request-config", help="Print an eval REQUEST_CONFIG bound to a seeded version")
    config.add_argument("seed", choices=["systemone", "openai-decisions"])
    config.add_argument("--url", required=True)
    config.add_argument("--model", default="")
    config.add_argument("--auth", default="none", choices=["none", "bearer"])

    args = parser.parse_args(argv)

    if args.command == "audit":
        _audit(args.path)
    elif args.adapters_command == "seed-sql":
        from decisioneval.transforms.adapter import seed_sql

        print(seed_sql(), end="")
    else:
        from decisioneval.transforms.adapter import seed_request_config

        print(json.dumps(seed_request_config(args.seed, args.url, args.model, {"kind": args.auth}), ensure_ascii=False))


def _audit(path: str) -> None:
    audit = audit_run_file(path)
    print(audit.report())
    print("FINAL: " + ("PASS" if audit.passed else "FAIL"))
    if not audit.passed:
        raise SystemExit(1)
