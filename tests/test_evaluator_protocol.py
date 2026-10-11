import json
import unittest
from unittest.mock import patch

import nodes


VALID_RESULT = {
    "score_a": 7.5,
    "score_b": 8.0,
    "selected": "B",
    "a": "Obraz A jest poprawny.",
    "b": "Obraz B lepiej pasuje do celu.",
    "keep": "Zachowac kompozycje.",
    "change": "Poprawic oswietlenie.",
    "improved_prompt": "A red cow standing in a grassy field",
    "comparison_pl": "",
}


class EvaluatorProtocolTests(unittest.TestCase):
    def test_empty_content_is_diagnosable(self):
        self.assertEqual(
            nodes.response_text({
                "choices": [{"message": {"content": "", "thinking": "reasoning"},
                             "finish_reason": "length"}],
                "usage": {"completion_tokens": 1024},
            }),
            "",
        )
        with self.assertRaisesRegex(RuntimeError, "empty content"):
            nodes.parse_json("  ")

    def test_native_ollama_vision_payload(self):
        data = "data:image/jpeg;base64,QUJD"
        payload = nodes.ollama_native_payload("model:latest", "system", "user", data, data)
        self.assertFalse(payload["stream"])
        self.assertFalse(payload["think"])
        self.assertEqual(payload["format"], "json")
        self.assertEqual(payload["messages"][1]["images"], ["QUJD", "QUJD"])
        self.assertEqual(
            nodes.ollama_native_endpoint("http://127.0.0.1:11434/v1/chat/completions"),
            "http://127.0.0.1:11434/api/chat",
        )
        self.assertIsNone(nodes.ollama_native_endpoint("http://localhost:1234/v1/models"))

    @patch.object(nodes, "free_cache")
    @patch.object(nodes, "unload_ollama_model")
    @patch.object(nodes, "image_to_data_url", return_value="data:image/jpeg;base64,QUJD")
    @patch.object(nodes, "post_json")
    def test_retry_after_openai_empty_content(self, request, image, unload, cache):
        request.side_effect = [
            {"choices": [{"message": {"content": ""},
                          "finish_reason": "length"}]},
            {"message": {"content": json.dumps(VALID_RESULT)},
             "done_reason": "stop"},
        ]
        evaluator = nodes.Krea2IterationEvaluator()
        result = evaluator.evaluate(
            None, None, "a cow", "photorealism",
            "http://127.0.0.1:11434/v1/chat/completions",
            "test-model", 60, True, True,
        )
        self.assertEqual(result[0], VALID_RESULT["improved_prompt"])
        self.assertEqual(result[4], "B")
        self.assertEqual(request.call_count, 2)
        self.assertEqual(request.call_args_list[1].args[0],
                         "http://127.0.0.1:11434/api/chat")
        unload.assert_called_once()
        cache.assert_called_once()

    @patch.object(nodes, "unload_ollama_model")
    @patch.object(nodes, "image_to_data_url", return_value="data:image/jpeg;base64,QUJD")
    @patch.object(nodes, "post_json")
    def test_both_empty_returns_error_and_unloads(self, request, image, unload):
        request.side_effect = [
            {"choices": [{"message": {"content": ""}}]},
            {"message": {"content": ""}, "done_reason": "length"},
        ]
        evaluator = nodes.Krea2IterationEvaluator()
        with self.assertRaisesRegex(RuntimeError, "after Ollama native retry"):
            evaluator.evaluate(
                None, None, "", "",
                "http://127.0.0.1:11434/v1/chat/completions",
                "test-model", 60, False, True,
            )
        self.assertEqual(request.call_count, 2)
        unload.assert_called_once()


if __name__ == "__main__":
    unittest.main()
