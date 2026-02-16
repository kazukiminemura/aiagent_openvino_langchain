from __future__ import annotations

import argparse

from app.agent.runner import MVPAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenVINO + LangChain MVP (document create focused)")
    parser.add_argument("--title", required=True, help="Document title")
    parser.add_argument("--content", required=True, help="Document content")
    parser.add_argument("--format", default="md", choices=["md", "txt"], help="Document format")
    parser.add_argument("--output-dir", default=None, help="Sub directory under allowed output root")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    agent = MVPAgent()
    result = agent.create_document(
        title=args.title,
        content=args.content,
        format=args.format,
        output_dir=args.output_dir,
    )
    print(result.message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
