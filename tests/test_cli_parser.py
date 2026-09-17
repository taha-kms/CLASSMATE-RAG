from pathlib import Path

import pytest

from cli.main import build_parser

SUBCOMMANDS = {
    "add",
    "ask",
    "preview",
    "stats",
    "dump",
    "restore",
    "vacuum",
    "rebuild",
    "list",
    "show",
    "delete",
    "reingest",
}


def _parse(argv):
    return build_parser().parse_args(argv)


def test_top_level_command_dispatch():
    # `command` is the public top-level dest set by add_subparsers(dest="command").
    assert _parse(["stats"]).command == "stats"
    assert _parse(["vacuum"]).command == "vacuum"


def test_all_subcommands_are_registered():
    # Parsing a minimal valid argv for each subcommand exercises registration
    # without poking argparse internals.
    minimal = {
        "add": ["add", "doc.pdf"],
        "ask": ["ask", "what?"],
        "preview": ["preview", "what?"],
        "stats": ["stats"],
        "dump": ["dump", "--path", "out.jsonl"],
        "restore": ["restore", "--path", "in.jsonl"],
        "vacuum": ["vacuum"],
        "rebuild": ["rebuild", "--model", "m"],
        "list": ["list"],
        "show": ["show", "--id", "x"],
        "delete": ["delete", "--id", "x"],
        "reingest": ["reingest", "--path", "doc.pdf"],
    }
    assert set(minimal) == SUBCOMMANDS
    for name, argv in minimal.items():
        ns = _parse(argv)
        assert ns.command == name


def test_add_flags_present():
    ns = _parse(
        [
            "add",
            "doc.pdf",
            "--course",
            "cs50",
            "--unit",
            "1",
            "--language",
            "en",
            "--doc-type",
            "pdf",
            "--author",
            "Alice",
            "--semester",
            "2025S",
            "--tags",
            "exam,week1",
        ]
    )
    assert ns.path == "doc.pdf"
    assert ns.course == "cs50"
    assert ns.unit == "1"
    assert ns.language == "en"
    assert ns.doc_type == "pdf"
    assert ns.author == "Alice"
    assert ns.semester == "2025S"
    assert ns.tags == "exam,week1"


def test_ask_flags_and_defaults():
    ns = _parse(["ask", "what?"])
    assert ns.question == "what?"
    assert ns.k == 8
    assert ns.hybrid == "on"
    assert ns.language == "auto"

    ns2 = _parse(["ask", "what?", "--k", "5", "--hybrid", "off"])
    assert ns2.k == 5
    assert ns2.hybrid == "off"


def test_doc_type_choices_include_epub():
    # Sanity check: epub is a valid choice for --doc-type.
    ns = _parse(["add", "book.epub", "--doc-type", "epub"])
    assert ns.doc_type == "epub"


def test_invalid_doc_type_rejected():
    with pytest.raises(SystemExit):
        _parse(["add", "x.pdf", "--doc-type", "pptzzz"])


def test_parser_prog_matches_the_installed_command_name():
    # The console script is named `rag`, so --help should say `rag`, not
    # whatever the module happens to be called.
    assert build_parser().prog == "rag"


def test_pyproject_declares_the_rag_console_script():
    # Guards the entry point against cli.main:main being moved or renamed
    # without pyproject.toml following it. Read as text so this still runs
    # on 3.10, which has no tomllib.
    root = Path(__file__).resolve().parents[1]
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'rag = "cli.main:main"' in text


def test_console_script_target_is_callable():
    from cli.main import main

    assert callable(main)
