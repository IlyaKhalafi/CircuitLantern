# Development

[Back to Circuit Lantern](../README.md)

```bash
make test
make check
make build
make test-ui
```

Browser verification uses a deterministic local GPT-2 fixture with real safetensors weights and a local tokenizer. It builds a temporary frontend snapshot with the installed dependencies, preserving running production builds and generated files in the working tree. It starts an isolated backend and a production Next.js server, then exercises model loading, graph navigation, weight statistics, traces, patching, persistence, exports, accessibility and mobile layouts. It needs no Hub account or model download. Install the browser once with `cd web && npx playwright install chromium`. Test services use ports 8011 and 3110 by default and refuse to reuse running services. Override `LANTERN_TEST_BACKEND_PORT` and `LANTERN_TEST_WEB_PORT` if needed.

CI runs the browser suite in two independent jobs. Each gets its own backend, model fixture, frontend build and evidence artifact. Test files stay together so state within a file remains sequential. Both jobs finish even if one fails, and the overall web check passes only when both succeed. To reproduce one partition locally, run `bash scripts/test-ui.sh --shard=1/2` or `bash scripts/test-ui.sh --shard=2/2`. Running `make test-ui` still checks the complete suite. Startup failures print their service log immediately.

The backend separates Hub access, hardware discovery, compatibility, model execution, numerical analysis and experiment storage into services. Model operations are serialized to keep hooks and weights consistent. Downloads run as background jobs. A failed download preserves the active model. After downloading, the loader repeats the hardware and storage checks under the model operation lock. If available resources have changed, it reports the failure and preserves the active model. Loading a replacement releases the previous model first to avoid holding two checkpoints in memory, so a later loading failure leaves no model active. Only safetensors checkpoints are accepted and remote model code is disabled.

Contributions should keep the researcher workflow focused, document measurement assumptions, and include checks for numerical or runtime changes. See [the architecture notes](architecture.md).
