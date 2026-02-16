from __future__ import annotations

import argparse
import json

from app.agent.runner import MVPAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenVINO + LangChain MVP Agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create a document")
    create_parser.add_argument("--title", required=True, help="Document title")
    create_parser.add_argument("--content", required=True, help="Document content")
    create_parser.add_argument("--format", default="md", choices=["md", "txt"], help="Document format")
    create_parser.add_argument("--output-dir", default=None, help="Sub directory under allowed output root")

    search_parser = subparsers.add_parser("search", help="Search files")
    search_parser.add_argument("--root-path", default=".", help="Search root under allowed output root")
    search_parser.add_argument("--pattern", default="*.md", help="Glob pattern")
    search_parser.add_argument("--max-results", type=int, default=20, help="Maximum number of results")

    chat_parser = subparsers.add_parser("chat", help="Auto-select tool from a natural language prompt")
    chat_parser.add_argument("--prompt", required=True, help="Natural language instruction")

    return parser


def main() -> int:
    args = build_parser().parse_args()
    agent = MVPAgent()

    if args.command == "create":
        result = agent.create_document(
            title=args.title,
            content=args.content,
            format=args.format,
            output_dir=args.output_dir,
        )
        print(result.message)
        print(json.dumps(result.data, ensure_ascii=False, indent=2))
        return 0

    if args.command == "search":
        result = agent.search_files(
            root_path=args.root_path,
            pattern=args.pattern,
            max_results=args.max_results,
        )
        print(result.message)
        print(json.dumps(result.data, ensure_ascii=False, indent=2))
        return 0

    if args.command == "chat":
        result = agent.run_prompt(args.prompt)
        print(result.message)
        print(json.dumps(result.data, ensure_ascii=False, indent=2))
        return 0

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
