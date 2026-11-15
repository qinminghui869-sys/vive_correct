import base64
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vive_correct.batch.codex_transport import CodexTransport
from vive_correct.estimate.openai_vision import OpenAIConfig, OpenAIError


class CodexTransportTests(unittest.TestCase):
    def test_preserves_images_order_and_exact_model(self):
        with TemporaryDirectory() as temp, patch("shutil.which", return_value="/bin/codex"):
            transport = CodexTransport(temp)
            config = OpenAIConfig("not-a-key", "codex://chatgpt-account", "gpt-6.1-sol", "login")
            body = dict(
                model=config.model,
                messages=[
                    dict(
                        content=[
                            dict(type="text", text="candidate_id=first"),
                            dict(
                                type="image_url",
                                image_url=dict(
                                    url="data:image/jpeg;base64,"
                                    + base64.b64encode(b"jpeg-one").decode()
                                ),
                            ),
                            dict(type="text", text="candidate_id=second"),
                            dict(
                                type="image_url",
                                image_url=dict(
                                    url="data:image/jpeg;base64,"
                                    + base64.b64encode(b"jpeg-two").decode()
                                ),
                            ),
                        ]
                    )
                ],
            )

            def run(command, **kwargs):
                self.assertEqual(command[command.index("--model") + 1], config.model)
                self.assertIn("--ephemeral", command)
                self.assertEqual(kwargs["input"].count("[附件图片"), 2)
                Path(command[command.index("-o") + 1]).write_text('{"decisions":[]}')
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch("subprocess.run", side_effect=run):
                self.assertEqual(transport(config, body), '{"decisions":[]}')
            directory = next(Path(temp).iterdir())
            self.assertEqual((directory / "image_01.jpg").read_bytes(), b"jpeg-one")
            self.assertEqual((directory / "image_02.jpg").read_bytes(), b"jpeg-two")
            self.assertEqual(
                json.loads((directory / "invocation.json").read_text())["model"], config.model
            )

            def bad_run(command, **kwargs):
                result = run(command, **kwargs)
                result.stdout = json.dumps(dict(item=dict(type="command_execution")))
                return result

            with patch("subprocess.run", side_effect=bad_run), self.assertRaises(OpenAIError):
                transport(config, body)

    def test_rejects_model_mismatch_before_request(self):
        with TemporaryDirectory() as temp, patch("shutil.which", return_value="/bin/codex"):
            transport = CodexTransport(temp)
            config = OpenAIConfig("unused", "", "another-model", "none")
            with patch("subprocess.run") as run, self.assertRaises(ValueError):
                transport(config, dict(model="another-model"))
            run.assert_not_called()
