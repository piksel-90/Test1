import base64
import gc
import json
import os
import urllib.request
import urllib.error
from urllib.parse import urlparse
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
    """Read text from OpenAI-compatible or Ollama-native responses."""
    if not isinstance(response, dict):
        raise RuntimeError("Evaluator returned a non-object response")
    if "choices" in response:
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError("Evaluator response has no choices")
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "\n".join(
                part.get("text", "") for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        if not isinstance(content, str):
            raise RuntimeError("Evaluator returned unsupported content type")
        if not content.strip():
            print("[Krea2 Evaluator] Empty OpenAI content; finish_reason="
                  + repr(choices[0].get("finish_reason"))
                  + ", thinking_chars=" + str(len(message.get("thinking") or ""))
                  + ", usage=" + repr(response.get("usage")))
        return content
    if "message" in response:
        message = response.get("message") or {}
        content = message.get("content") or ""
        if not isinstance(content, str):
            raise RuntimeError("Ollama returned unsupported content type")
        if not content.strip():
            print("[Krea2 Evaluator] Empty Ollama content; done_reason="
                  + repr(response.get("done_reason"))
                  + ", thinking_chars=" + str(len(message.get("thinking") or ""))
                  + ", eval_count=" + repr(response.get("eval_count")))
        return content
    raise RuntimeError("Evaluator response has neither choices nor message")


def ollama_native_endpoint(endpoint):
    """Recognize Ollama's OpenAI-compatible route, including remote hosts."""
    parsed = urlparse(endpoint.strip())
    if (parsed.scheme in ("http", "https")
            and parsed.netloc
            and parsed.path.rstrip("/") == "/v1/chat/completions"):
        return parsed.scheme + "://" + parsed.netloc + "/api/chat"
    return None


def ollama_native_payload(model, system, user, data_a, data_b):
    """Ollama's native vision API expects raw base64 strings, not data URLs."""
    return {
        "model": model,
        "stream": False,
        "format": "json",
        "think": False,
        "options": {"temperature": 0.2, "num_predict": 4096},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",
             "content": user + "\n\nCompare the two attached images in order: first A, then B. Return the JSON evaluation.",
             "images": [data_a.split(",", 1)[1], data_b.split(",", 1)[1]]},
        ],
    }

def parse_json(text):
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Evaluator returned empty content instead of JSON")
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

def unload_ollama_model(endpoint, model, timeout):
    """Ask Ollama to unload the evaluator model immediately."""
    try:
        parsed = urlparse(endpoint.strip())
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return
        base = parsed.scheme + "://" + parsed.netloc
        unload_url = base + "/api/generate"
        payload = {"model": model, "prompt": "", "keep_alive": 0}
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            unload_url, data=body,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=min(int(timeout), 30)) as response:
            response.read()
        print("[Krea2 Evaluator] Ollama model unloaded: " + str(model))
    except Exception as e:
        print("[Krea2 Evaluator] WARNING: could not unload Ollama model '" + str(model) + "': " + str(e))


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
                "unload_ollama": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "extra_instruction": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("improved_prompt", "evaluation_json", "score_a", "score_b", "selected", "comparison_pl")
    FUNCTION = "evaluate"
    CATEGORY = "Krea 2 / Iteration"

    def evaluate(self, image_a, image_b, original_prompt, target, endpoint, model, timeout, free_cuda_cache, unload_ollama, extra_instruction=""):
        data_a = image_to_data_url(image_a)
        data_b = image_to_data_url(image_b)
        system = """You are a strict visual evaluator and prompt optimizer for Krea 2 Turbo.

You receive ORIGINAL PROMPT, TARGET/REQUIREMENTS, and two candidate images A and B.
Judge the images primarily by how well they satisfy the ORIGINAL PROMPT and TARGET,
not merely by which image looks prettier in isolation.

Compare A and B carefully. Identify concrete strengths and weaknesses in each image.
Select the better candidate. Explain what should be preserved and what should be changed.
Then write an improved, production-ready Krea 2 prompt for the NEXT iteration.
Preserve successful elements from the selected candidate and change only material problems.
Do not invent requirements that are absent from the ORIGINAL PROMPT or TARGET.

IMPORTANT LANGUAGE RULE:
All descriptive text fields MUST be written in Polish:
- a
- b
- keep
- change
- comparison_pl
The improved_prompt should remain in the language that is most effective for Krea 2 and should preserve the language of the ORIGINAL PROMPT when practical.

comparison_pl must be a detailed Polish report, not a one-line summary. It must explain the result and the next iteration clearly.

IMPORTANT OUTPUT RULE:
Return ONLY one valid JSON object.
Do NOT use Markdown fences.
Do NOT write an explanation before or after the JSON.

Required JSON schema:
{
  "score_a": 0.0,
  "score_b": 0.0,
  "selected": "A",
  "a": "szczegółowa analiza obrazu A po polsku: zgodność z promptem, kompozycja, anatomia, światło, materiały, szczegóły, artefakty i konkretne mocne/słabe strony",
  "b": "szczegółowa analiza obrazu B po polsku: zgodność z promptem, kompozycja, anatomia, światło, materiały, szczegóły, artefakty i konkretne mocne/słabe strony",
  "keep": "szczegółowo po polsku: co zachować z najlepszego wyniku i dlaczego",
  "change": "szczegółowo po polsku: co konkretnie zmienić w następnej iteracji i jaki problem każda zmiana ma rozwiązać",
  "improved_prompt": "complete production-ready Krea 2 prompt",
  "comparison_pl": "pełny raport po polsku w formacie: PORÓWNANIE KANDYDATÓW / LEPSZY OBRAZ / WYNIK A / WYNIK B / ANALIZA A / ANALIZA B / CO ZACHOWAĆ / CO ZMIENIĆ / PROMPT NASTĘPNEJ ITERACJI"
}

score_a and score_b must be numbers from 0 to 10.
selected must be exactly A or B.
All JSON values must be valid JSON strings/numbers."""
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
        response = None
        try:
            response = post_json(endpoint.strip(), payload, int(timeout))
            raw_response = response_text(response)
            try:
                result = parse_json(raw_response)
            except (RuntimeError, ValueError, TypeError) as first_error:
                native_url = ollama_native_endpoint(endpoint)
                if not native_url:
                    raise RuntimeError(
                        "Evaluator returned invalid JSON: " + str(first_error)
                        + "; response excerpt: " + repr(str(raw_response)[:600])
                    ) from first_error
                print("[Krea2 Evaluator] OpenAI-compatible response not usable: "
                      + str(first_error))
                print("[Krea2 Evaluator] Retrying via Ollama /api/chat "
                      "with think=false and format=json")
                native_payload = ollama_native_payload(
                    model, system, user, data_a, data_b
                )
                try:
                    response = post_json(native_url, native_payload, int(timeout))
                    raw_response = response_text(response)
                    result = parse_json(raw_response)
                except (RuntimeError, ValueError, TypeError) as retry_error:
                    raise RuntimeError(
                        "Evaluator failed after Ollama native retry. "
                        "Initial response: " + str(first_error)
                        + "; retry error: " + str(retry_error)
                        + "; response excerpt: " + repr(str(raw_response)[:600])
                    ) from retry_error
            print("[Krea2 Evaluator] Parsed evaluation JSON successfully.")
        finally:
            if unload_ollama:
                unload_ollama_model(endpoint, model, int(timeout))
        score_a = max(0.0, min(10.0, float(result.get("score_a", 0.0))))
        score_b = max(0.0, min(10.0, float(result.get("score_b", 0.0))))
        improved = str(result.get("improved_prompt", "")).strip()
        if not improved:
            raise RuntimeError("Evaluator returned an empty improved_prompt")
        selected = str(result.get("selected", "MIX")).upper()
        if selected not in ("A", "B", "MIX"):
            selected = "MIX"
        # Build a detailed Polish report from the SAME structured evaluation.
        # This keeps comparison_pl consistent with evaluation_json instead of making
        # it a separate, lossy one-line summary.
        a_text = str(result.get("a", "Brak szczegółowej analizy obrazu A.")).strip()
        b_text = str(result.get("b", "Brak szczegółowej analizy obrazu B.")).strip()
        keep_text = str(result.get("keep", "Brak danych dotyczących elementów do zachowania.")).strip()
        change_text = str(result.get("change", "Brak danych dotyczących zmian.")).strip()
        model_comparison = str(result.get("comparison_pl", "")).strip()
        comparison_pl = (
            "PORÓWNANIE KANDYDATÓW\n\n"
            "LEPSZY OBRAZ: " + selected + "\n"
            "WYNIK A: " + f"{score_a:.1f}" + "/10\n"
            "WYNIK B: " + f"{score_b:.1f}" + "/10\n\n"
            "ANALIZA A:\n" + a_text + "\n\n"
            "ANALIZA B:\n" + b_text + "\n\n"
            "CO ZACHOWAĆ:\n" + keep_text + "\n\n"
            "CO ZMIENIĆ:\n" + change_text + "\n\n"
            "PROMPT NASTĘPNEJ ITERACJI:\n" + improved
        )
        if model_comparison and len(model_comparison) > 80:
            # Preserve a rich model-written Polish report when it actually contains
            # useful detail, but still ensure the output is never a terse one-liner.
            comparison_pl = model_comparison + "\n\n" + (
                "WNIOSKI TECHNICZNE:\n"
                "Do kolejnej iteracji należy zachować elementy wskazane w sekcji "
                "'CO ZACHOWAĆ' i zastosować wyłącznie istotne poprawki z sekcji "
                "'CO ZMIENIĆ'."
            )

        # Ensure the machine-readable JSON contains the same detailed Polish report.
        result["comparison_pl"] = comparison_pl
        evaluation = json.dumps(result, ensure_ascii=False, indent=2)
        if free_cuda_cache:
            del data_a, data_b, response
            free_cache()
        return improved, evaluation, score_a, score_b, selected, comparison_pl

NODE_CLASS_MAPPINGS = {"Krea2IterationEvaluator": Krea2IterationEvaluator}
NODE_DISPLAY_NAME_MAPPINGS = {"Krea2IterationEvaluator": "Krea 2 Iteration Evaluator (2 Images)"}
