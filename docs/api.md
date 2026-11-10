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

## Host check

On a loopback bind (the default), requests whose `Host` header is not `localhost`, `127.0.0.1` or `::1` get
`400`. This turns away a web page that uses DNS rebinding to make your browser call the API as if it were
the page's own origin: that page has to send its own hostname as `Host`. Open the UI as
`http://localhost:8080` or `http://127.0.0.1:8080`; `http://0.0.0.0:8080` is refused on purpose.

With `--host` set to anything else the names people use are not knowable, so the check is off and the
warning above applies. Behind a Docker port-forward that means no check; keep the published port on
`127.0.0.1`. This does not protect against another user on the same machine, who can reach loopback
legitimately.

## Endpoints

| Method | Path | Does |
| --- | --- | --- |
| `POST` | `/ask` | Answer a question. Server-Sent Events, streamed as produced. |
| `POST` | `/preview` | Retrieval only, with scores. No model is loaded. |
| `POST` | `/ingest` | Upload and ingest a file. Fields mirror `rag add`. |
| `GET` | `/chunks` | List indexed chunks, same filters as `rag list`. |
| `GET` | `/facets` | The values already in the corpus for `course`, `unit`, `author` and `semester` (sorted, at most 200 each), which fill the web UI's filter dropdowns. |
| `GET` | `/chunks/{id}` | One chunk, as `rag show`. |
| `DELETE` | `/chunks` | Delete by ids, path or filter. Dry run by default. |
| `GET` | `/stats` | Index health, as `rag stats`. |
| `GET` | `/reconcile` | Chunks stranded in one store, as `rag reconcile`. |
| `GET` | `/settings` | Every setting, its effective value and where that came from. Credentials are masked. |
| `PUT` | `/settings` | Change settings together, as `rag config set`. All or nothing. |
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

### Settings

`GET /settings` lists every setting in the same registry `rag config list` uses:

```json
{"key": "K_VECTOR", "kind": "int", "group": "Retrieval", "value": "8", "source": "file",
 "default": "8", "minimum": 1, "maximum": 100, "restart_required": false,
 "overridden": false, "file_value": null, "choices": [], "description": "..."}
```

`source` is `environment`, `file`, `stored` (a credential in the credential store), `default` or
`unset`. When `overridden` is true the shell holds a different value than `.env`, so saving to the
file changes nothing until that variable is unset; `file_value` is what the file says.

`PUT /settings` takes `{"changes": {"K_VECTOR": 12, "OPENAI_API_KEY": "..."}}`. A `null` value removes
a setting. Every change is checked first and nothing is written if any is invalid (`422`, with the
reason in `detail`). Each result says whether it `applied` and whether it needs a restart, and
carries a `note` when it did not take effect:

```json
{"changes": [{"key": "ENABLE_ROUTING", "action": "set", "applied": false,
              "restart_required": true, "note": "Takes effect after a restart."}]}
```

Settings that decide where credentials and text are sent, or what gets loaded, are read-only over HTTP
(`local_only: true` in the listing; `PUT` answers `422`): `OPENAI_BASE_URL`, `CHROMA_HTTP_URL`,
`EMBEDDING_MODEL_NAME`, every `*_MODEL_PATH`, and the index directories. Changing them needs `rag config set` on
this machine. `POST /ingest` keeps an upload under `data/uploads/`, so a writable model path would let a caller
make the server load a file they sent.

A setting marked `restart_required` is saved to `.env` but **not applied to the running server**, which could
otherwise load several large models mid-session. The listing keeps `value` as what the server is using and
puts the saved value in `pending_value` until it restarts. URL-valued settings are returned rebuilt from scheme,
host, port and path only (no userinfo, query or fragment), or hidden if they cannot be parsed.

Credentials are write-only: they are accepted here and stored outside `.env`, and read back only as
a masked form ending in the last four characters.

`POST /settings/validate` takes `{"provider": "anthropic" | "openai" | "gemini"}` and asks that provider's
read-only model-list endpoint whether the key the server would use is accepted. No key is sent in the request
and the provider must be one of the three names, so it cannot be aimed at another host. It tests the **effective**
key, the environment before the credential store, and says which one:

```json
{"provider": "openai", "outcome": "valid", "ok": true, "message": "The provider accepted the key.",
 "source": "stored", "masked": "********1234", "target": "https://api.openai.com/v1/models", "status": 200}
```

`outcome` is one of `valid`, `rejected` (401/403, and 400 from Gemini), `rate_limited`, `unreachable`,
`unexpected` or `missing` (no request is made). The provider's response body and any error text are never
returned. The key travels in a header, and redirects are not followed, so it is never re-sent elsewhere. With
`OPENAI_BASE_URL` set, `target` shows where the request went: a local server often accepts any key, so `valid`
there means that server answered, not that OpenAI would. The check is on demand only; saving a key does not
trigger it. From a terminal: `rag config test openai` (exit status 1 unless every key given is valid).

**This endpoint widens what an unauthenticated API can do.** Anyone who can reach the port can set
`OPENAI_BASE_URL` and `LLM_PROVIDER` and so redirect later questions, and the course text sent with
them, to a server of their choosing. That is the same "the bind address is the whole security model"
rule as above, now with more behind it.

## Serving the frontend

If `web/dist/index.html` exists it is mounted at `/`, so one process serves
both. It is mounted last and cannot shadow an API route. When it is absent the
API starts anyway and says so in the log — which is the state of every machine
before the first `npm run build`.
