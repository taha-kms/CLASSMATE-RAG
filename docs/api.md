# The local API

`rag serve` puts an HTTP API in front of the same pipeline the CLI uses, and
serves the web frontend from the same process when it has been built.

```bash
pip install -e ".[api]"
rag serve
```

That listens on <http://127.0.0.1:8080>. Interactive documentation, generated
from the route definitions, is at `/docs`.

## There is no authentication

None, by design. Every student runs their own copy against their own corpus on
their own machine, so an account system would be ceremony around a single user.

The consequence is that **the bind address is the entire security model**.
`rag serve` defaults to `127.0.0.1`, and `DELETE /chunks` and everything under
`/admin` are reachable by anyone who can reach the port. `--host` exists
because port-forwarding into a container needs it, and it prints a warning when
given anything other than loopback. Do not put this on a network you share.

## Endpoints

| Method | Path | Does |
| --- | --- | --- |
| `POST` | `/ask` | Answer a question. Server-Sent Events, streamed as produced. |
| `POST` | `/preview` | Retrieval only, with scores. No model is loaded. |
| `POST` | `/ingest` | Upload and ingest a file. Fields mirror `rag add`. |
| `GET` | `/chunks` | List indexed chunks, same filters as `rag list`. |
| `GET` | `/chunks/{id}` | One chunk, as `rag show`. |
| `DELETE` | `/chunks` | Delete by ids, path or filter. Dry run by default. |
| `GET` | `/stats` | Index health, as `rag stats`. |
| `GET` | `/reconcile` | Chunks stranded in one store, as `rag reconcile`. |
| `POST` | `/admin/vacuum` | Housekeeping. Fast enough to answer inline. |
| `POST` | `/admin/dump`, `/admin/restore`, `/admin/rebuild` | Start a job, return `202` and its id. |
| `GET` | `/admin/jobs`, `/admin/jobs/{id}` | Job state and result. |

A filter left out of a request body means "do not filter on this". It does not
mean "match documents whose course is null".

### Asking a question

```bash
curl -N -X POST http://127.0.0.1:8080/ask \
  -H 'content-type: application/json' \
  -d '{"question": "What is the chain rule?", "course": "Maths", "top_k": 5}'
```

Each event is named, so a browser can use `addEventListener` rather than
switching on a field:

| Event | Payload | Meaning |
| --- | --- | --- |
| `stage` | `{stage, message}` | Progress. `message` is written for display. |
| `token` | `{text}` | Append to what is on screen. |
| `replace` | `{reason}` | Discard everything so far; what follows supersedes it. |
| `final` | the full `AskResult` | Always last. Authoritative. |
| `error` | `{error}` | Something failed after the response had started. |

`final.answer` can differ from the concatenated tokens: citation enforcement
and translate-on-miss both rewrite the answer after generation. Render the
tokens as they arrive, then replace with `final.answer`.

An `error` event exists because the status code is already sent by the time
generation fails. Without it the client would see a stream that simply stops.

### Waiting for the model

Only one answer is generated at a time. The model is large, one copy is
resident, and two generations at once means two copies. A second request
therefore waits.

Rather than going quiet, it emits a `waiting` stage:

```
stage  retrieving     Searching your documents
stage  waiting        Waiting for another answer to finish
stage  loading_model  Loading the model
stage  generating     Writing the answer
```

Retrieval still runs immediately; only generation queues. Show the `waiting`
message, because otherwise the wait is indistinguishable from a hang.

### Uploading

```bash
curl -X POST 'http://127.0.0.1:8080/ingest?course=Maths&unit=3' \
  -F 'file=@notes.pdf'
```

The filename needs an extension: the loader and the document type are both
chosen from it, and a file without one is rejected rather than parsed as plain
text. Uploads are kept under `data/uploads/`, not streamed through a temporary
file, because `source_path` goes into every chunk's metadata and
`rag reingest`, `rag list --path` and `rag delete --path` all read it back.

### Long operations

`restore` and `rebuild` re-embed or rewrite the whole corpus, which takes
minutes. They return `202` with a job id immediately:

```bash
id=$(curl -sX POST http://127.0.0.1:8080/admin/rebuild \
      -H 'content-type: application/json' \
      -d '{"model": "intfloat/multilingual-e5-large"}' | jq -r .id)
curl -s "http://127.0.0.1:8080/admin/jobs/$id"
```

Jobs live in memory and die with the process. That is deliberate for a
single-user local application: a rebuild interrupted by a restart has to be
started again anyway, and persisting the record would only produce jobs that
claim to be running with nothing behind them.

### Deleting

`DELETE /chunks` is a dry run unless you say otherwise, and reports what it
would remove:

```bash
curl -X DELETE http://127.0.0.1:8080/chunks \
  -H 'content-type: application/json' \
  -d '{"course": "Maths", "dry_run": false}'
```

## Serving the frontend

If `web/dist/index.html` exists it is mounted at `/`, so one process serves
both. It is mounted last and cannot shadow an API route. When it is absent the
API starts anyway and says so in the log — which is the state of every machine
before the first `npm run build`.
