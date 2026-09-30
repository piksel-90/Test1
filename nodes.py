import base64
import gc
import json
import os
import urllib.request
import urllib.error
from io import BytesIO
import torch
from PIL import Image

def image_to_data_url(image):
    if image.ndim == 4:
        image = image[0]
    cpu = image.detach().to("cpu", dtype=torch.float32).clamp(0, 1)
    arr = (cpu.numpy() * 255.0 + 0.5).astype("uint8")
    pil = Image.fromarray(arr, "RGB")
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    del pil, cpu, arr
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def post_json(url, payload, timeout):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    key = os.getenv("KREA_EVAL_API_KEY", "").strip()
    if key:
        req.add_header("Authorization", "Bearer " + key)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError("Evaluator HTTP " + str(e.code) + ": " + e.read().decode("utf-8", errors="replace"))
    except urllib.error.URLError as e:
        raise RuntimeError("Cannot reach evaluator: " + str(e))

def response_text(response):
    try:
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError("Evaluator response is not OpenAI-compatible") from e

def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:] if lines else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("Evaluator did not return JSON")
    return json.loads(text[start:end + 1])

def free_cache():
    gc.collect()
    try:
        import comfy.model_management as mm
        mm.soft_empty_cache()
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass

class Krea2IterationEvaluator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "original_prompt": ("STRING", {"multiline": True, "default": ""}),
                "target": ("STRING", {"multiline": True, "default": ""}),
                "endpoint": ("STRING", {"default": "http://127.0.0.1:11434/v1/chat/completions"}),
                "model": ("STRING", {"default": "gemma3:4b"}),
                "timeout": ("INT", {"default": 180, "min": 10, "max": 3600, "step": 10}),
                "free_cuda_cache": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "extra_instruction": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("improved_prompt", "evaluation_json", "score_a", "score_b", "selected")
    FUNCTION = "evaluate"
    CATEGORY = "Krea 2 / Iteration"

    def evaluate(self, image_a, image_b, original_prompt, target, endpoint, model, timeout, free_cuda_cache, extra_instruction=""):
        data_a = image_to_data_url(image_a)
        data_b = image_to_data_url(image_b)
        system = """You are a strict visual evaluator and prompt refiner for Krea 2 Turbo.\nCompare candidate A and B against TARGET and ORIGINAL PROMPT.\nReturn only JSON with score_a, score_b, selected, a, b, keep, change and improved_prompt.\nScores are 0-10. Preserve what already works. Correct only material problems.\nThe improved_prompt must be one production-ready Krea 2 prompt."""
        user = "TARGET:\n" + (target or "(none)") + "\n\nORIGINAL PROMPT:\n" + (original_prompt or "(none)") + "\n\nEXTRA:\n" + (extra_instruction or "(none)")
        payload = {
            "model": model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": user},
                    {"type": "text", "text": "CANDIDATE A"},
                    {"type": "image_url", "image_url": {"url": data_a}},
                    {"type": "text", "text": "CANDIDATE B"},
                    {"type": "image_url", "image_url": {"url": data_b}}
                ]}
            ],
        }
        response = post_json(endpoint.strip(), payload, int(timeout))
        result = parse_json(response_text(response))
        score_a = max(0.0, min(10.0, float(result.get("score_a", 0.0))))
        score_b = max(0.0, min(10.0, float(result.get("score_b", 0.0))))
        improved = str(result.get("improved_prompt", "")).strip()
        if not improved:
            raise RuntimeError("Evaluator returned an empty improved_prompt")
        selected = str(result.get("selected", "MIX")).upper()
        if selected not in ("A", "B", "MIX"):
            selected = "MIX"
        evaluation = json.dumps(result, ensure_ascii=False, indent=2)
        if free_cuda_cache:
            del data_a, data_b, response
            free_cache()
        return improved, evaluation, score_a, score_b, selected

NODE_CLASS_MAPPINGS = {"Krea2IterationEvaluator": Krea2IterationEvaluator}
NODE_DISPLAY_NAME_MAPPINGS = {"Krea2IterationEvaluator": "Krea 2 Iteration Evaluator (2 Images)"}
