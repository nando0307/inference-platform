import json
import unittest

from pydantic import ValidationError

from src.api import ChatCompletionRequest
from src.inference import MODEL_ID


def request_data(**changes):
    return {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Hello"}],
        **changes,
    }


class RequestValidationTests(unittest.TestCase):
    def assert_rejected(self, data, code=None):
        # Exercise both Python callers and the JSON boundary used by HTTP clients.
        for source in (data, json.dumps(data)):
            with self.subTest(source_type=type(source).__name__):
                with self.assertRaises(ValidationError) as caught:
                    if isinstance(source, str):
                        ChatCompletionRequest.model_validate_json(source)
                    else:
                        ChatCompletionRequest.model_validate(source)
                if code is not None:
                    codes = {
                        error["type"]
                        for error in caught.exception.errors(include_input=False)
                    }
                    self.assertIn(code, codes)

    def test_valid_history_preserves_text_and_allows_consecutive_roles(self):
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "  Hello\n世界!  "},
            {"role": "user", "content": "Another question."},
            {"role": "assistant", "content": "An earlier answer."},
        ]
        result = ChatCompletionRequest.model_validate_json(
            json.dumps(request_data(messages=messages))
        )
        self.assertEqual([message.model_dump() for message in result.messages], messages)

    def test_valid_numeric_boundaries_and_integer_sampling_values(self):
        for settings in (
            {"max_tokens": 1, "temperature": 0, "top_p": 1},
            {"max_tokens": 512, "temperature": 2, "top_p": 1},
            {"temperature": 1, "top_p": 0.01},
            {"temperature": 0.7, "top_p": 0.9},
        ):
            with self.subTest(settings=settings):
                result = ChatCompletionRequest.model_validate_json(
                    json.dumps(request_data(**settings))
                )
                for field, value in settings.items():
                    self.assertEqual(getattr(result, field), value)

    def test_required_fields_and_model_identity(self):
        for field in ("model", "messages"):
            data = request_data()
            del data[field]
            self.assert_rejected(data, "missing")
        for model in ("another-model", "", 123, None):
            self.assert_rejected(request_data(model=model))

    def test_invalid_messages_and_content(self):
        for messages in (
            [],
            "Hello",
            None,
            [{"role": "tool", "content": "Hello"}],
            [{"role": "user"}],
            [{"content": "Hello"}],
            [{"role": "user", "content": ""}],
            [{"role": "user", "content": " \t\n\u2003"}],
            [{"role": "user", "content": 123}],
            [{"role": "user", "content": [{"type": "image_url"}]}],
        ):
            with self.subTest(messages=messages):
                self.assert_rejected(request_data(messages=messages))

    def test_unknown_request_and_message_fields(self):
        for field in ("tools", "n", "top_k", "stop", "seed", "stream_options", "unknown"):
            with self.subTest(field=field):
                self.assert_rejected(request_data(**{field: None}), "extra_forbidden")
        self.assert_rejected(
            request_data(messages=[{"role": "user", "content": "Hello", "name": "x"}]),
            "extra_forbidden",
        )

    def test_numeric_types_do_not_coerce_strings_or_booleans(self):
        for field in ("max_tokens", "temperature", "top_p"):
            for value in ("1", True, False, None):
                with self.subTest(field=field, value=value):
                    self.assert_rejected(request_data(**{field: value}))
        self.assert_rejected(request_data(max_tokens=1.0), "int_type")

    def test_numeric_ranges(self):
        for field, values in (
            ("max_tokens", (0, -1, 513)),
            ("temperature", (-0.1, 2.01)),
            ("top_p", (0, -0.1, 1.01)),
        ):
            for value in values:
                with self.subTest(field=field, value=value):
                    self.assert_rejected(request_data(**{field: value}))

    def test_nonfinite_numbers(self):
        for field in ("max_tokens", "temperature", "top_p"):
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(field=field, value=value):
                    self.assert_rejected(request_data(**{field: value}))

    def test_stream_requires_boolean_and_rejects_unsupported_streaming(self):
        for value in ("false", "true", 0, 1, None):
            with self.subTest(value=value):
                self.assert_rejected(request_data(stream=value), "bool_type")
        self.assert_rejected(request_data(stream=True), "unsupported_parameter")

    def test_greedy_top_p_combination_including_default_temperature(self):
        self.assert_rejected(request_data(top_p=0.9), "invalid_parameter_combination")
        self.assert_rejected(
            request_data(temperature=0, top_p=0.9), "invalid_parameter_combination"
        )

    def test_malformed_json_and_nonobject_body(self):
        for body in ('{"model":', "[]", "null", '"text"'):
            with self.subTest(body=body):
                with self.assertRaises(ValidationError):
                    ChatCompletionRequest.model_validate_json(body)


if __name__ == "__main__":
    unittest.main()
