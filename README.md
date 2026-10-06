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
