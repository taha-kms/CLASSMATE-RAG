# Branch rules

`main.json` is the ruleset that protects `main`, kept here so it can be reviewed and re-created. GitHub does not
read this file; it is applied with:

```bash
gh api -X PUT repos/OWNER/REPO/rulesets/RULESET_ID --input .github/rulesets/main.json
# or, on a fresh repository, POST to repos/OWNER/REPO/rulesets
```

What it enforces:

- changes to `main` come through a pull request, with all review threads resolved
- these checks must pass before merging: `pytest`, `Typecheck, lint and format`, `End-to-end`,
  `Build and smoke test the image`, `Scan for committed secrets`, `Analyze Python`, `CodeQL`
- no deleting `main`, no force-pushing it

Left out on purpose:

- **SonarCloud** and the **Trivy** results check. A third-party outage should not block a merge. The Trivy
  *gate* still runs inside "Build and smoke test the image" and fails it on a new CRITICAL finding.
- **Up-to-date branches.** Requiring every PR to be rebased onto the latest `main` adds friction for stacked
  PRs and gains little on a repository with one regular contributor.

Repository admins can still bypass the ruleset, through a pull request only. That is the escape hatch for a
broken check, not the normal way to merge: a green PR merges with a plain `gh pr merge`.

If a required check is renamed in a workflow, update `main.json` and re-apply it, or every PR will wait for a
check that no longer exists.
