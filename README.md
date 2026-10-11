# Krea 2 Iteration Evaluator

ComfyUI custom node for the loop:

Krea 2 Turbo -> 2 images -> visual evaluation -> improved prompt -> next iteration.

## No QwenVL-Mod dependency

This node does not import QwenVL-Mod, AILab_QwenVL, or any ComfyUI vision node pack.
It sends the two candidate images to an OpenAI-compatible multimodal endpoint.

Default endpoint:
http://127.0.0.1:11434/v1/chat/completions

Default model:
gemma3:4b

Choose your installed Ollama vision model in the node (for example,
`lukey03/qwen3.5-9b-abliterated-vision:latest`). This is a local
HTTP connection; images are not sent to a cloud service.

The endpoint can be Ollama, LM Studio, vLLM, or another compatible server.
Set KREA_EVAL_API_KEY when the endpoint requires Bearer authentication.

## Outputs

- improved_prompt
- evaluation_json
- score_a
- score_b
- selected (A/B/MIX)

## VRAM behavior

Images are converted to JPEG on CPU before the HTTP request. After evaluation
the node calls ComfyUI soft_empty_cache() and PyTorch CUDA cache cleanup.

This cannot forcibly unload a Krea model owned by an upstream node. Reliable
staged VRAM usage requires generation and evaluation to be separate queue stages
or an explicit model unload between them.

## Install

Copy this repository into:
ComfyUI/custom_nodes/Test1

Restart ComfyUI.

Node category:
Krea 2 / Iteration


## Empty / invalid Ollama JSON responses

Some Ollama vision models can return HTTP 200 with empty `message.content`
(or use their output budget on thinking). Instead of crashing immediately,
the evaluator now logs the completion status and retries once using Ollama's
native `/api/chat` endpoint with `think: false`, `format: "json"`, and
`stream: false`. This fallback applies when the configured URL ends in
`/v1/chat/completions`. The model is unloaded **after** both attempts,
not between them, when `unload_ollama` is enabled.

If both attempts fail, the error includes the initial failure, retry error,
and a short response excerpt. To debug without repeatedly reloading the
vision model, temporarily disable `unload_ollama` and inspect the ComfyUI log.

Update the node by pulling branch
`feature/krea2-iteration-evaluator`, then restart ComfyUI.
Regression tests (with the ComfyUI Python environment):

```sh
python -m unittest discover -s tests -v
```

These tests mock Ollama requests. They do not replace a live test with your
specific installed vision model.
