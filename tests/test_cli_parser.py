from cli.main import build_parser

def test_cli_has_expected_subcommands_and_flags():
    p = build_parser()
    subcommands = {sp.dest for sp in p._subparsers._group_actions[0].choices.values()}
    # we can also just assert names directly
    choices = set(p._subparsers._group_actions[0].choices.keys())
    for name in {"add", "ask", "preview", "stats", "dump", "restore", "vacuum", "rebuild", "list", "show", "delete", "reingest"}:
        assert name in choices

    # spot-check a couple of flags on "add" and "ask"
    add = p._subparsers._group_actions[0].choices["add"]
    add_flags = {a.dest for a in add._actions}
    for flag in {"path", "course", "unit", "language", "doc_type", "author", "semester", "tags"}:
        assert flag in add_flags

    ask = p._subparsers._group_actions[0].choices["ask"]
    ask_flags = {a.dest for a in ask._actions}
    for flag in {"question", "k", "hybrid"}:
        assert flag in ask_flags
