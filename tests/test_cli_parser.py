import pytest

from cli.main import build_parser


SUBCOMMANDS = {
    "add", "ask", "preview", "stats", "dump", "restore", "vacuum",
    "rebuild", "list", "show", "delete", "reingest",
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
    ns = _parse([
        "add", "doc.pdf",
        "--course", "cs50",
        "--unit", "1",
        "--language", "en",
        "--doc-type", "pdf",
        "--author", "Alice",
        "--semester", "2025S",
        "--tags", "exam,week1",
    ])
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
