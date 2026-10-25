import unittest
from unittest.mock import patch, MagicMock, Mock
import sys
import os
import json
import base64

# handler.py lives at the repository root; it imports network_volume as a
# sibling module (both are ADDed to / in the Docker image), which lives in
# src/ in the repository — so both directories must be importable.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, _REPO_ROOT)
import handler


def _make_object_info():
    """A minimal /object_info payload covering the loaders under test."""
    return {
        "CheckpointLoaderSimple": {
            "input": {
                "required": {
                    "ckpt_name": [
                        ["sd_xl_base_1.0.safetensors", "subdir/anime_v2.safetensors"],
                        {},
                    ]
                }
            }
        },
        "LoraLoader": {
            "input": {
                "required": {
                    "lora_name": [["detail_tweaker.safetensors"], {}],
                    "model": ["MODEL"],
                    "clip": ["CLIP"],
                }
            }
        },
        "VAELoader": {
            "input": {"required": {"vae_name": [["sdxl_vae.safetensors"], {}]}}
        },
        "DualCLIPLoader": {
            "input": {
                "required": {
                    "clip_name1": [["clip_l.safetensors"], {}],
                    "clip_name2": [["t5xxl_fp16.safetensors"], {}],
                }
            }
        },
        "UNETLoader": {
            "input": {"required": {"unet_name": [["flux1-dev.safetensors"], {}]}}
        },
        "UpscaleModelLoader": {
            "input": {"required": {"model_name": [["4x_ultrasharp.pth"], {}]}}
        },
        "LoadImage": {
            "input": {"required": {"image": [["example.png"], {}]}}
        },
    }


def _mock_object_info_response(object_info):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = object_info
    return mock_response


class TestValidateInput(unittest.TestCase):
    def test_valid_input_with_workflow_only(self):
        input_data = {"workflow": {"key": "value"}}
        validated_data, error = handler.validate_input(input_data)
        self.assertIsNone(error)
        self.assertEqual(
            validated_data,
            {"workflow": {"key": "value"}, "images": None, "comfy_org_api_key": None},
        )

    def test_valid_input_with_workflow_and_images(self):
        input_data = {
            "workflow": {"key": "value"},
            "images": [{"name": "image1.png", "image": "base64string"}],
        }
        validated_data, error = handler.validate_input(input_data)
        self.assertIsNone(error)
        self.assertEqual(validated_data["workflow"], input_data["workflow"])
        self.assertEqual(validated_data["images"], input_data["images"])

    def test_input_missing_workflow(self):
        input_data = {"images": [{"name": "image1.png", "image": "base64string"}]}
        validated_data, error = handler.validate_input(input_data)
        self.assertIsNotNone(error)
        self.assertEqual(error, "Missing 'workflow' parameter")

    def test_input_with_invalid_images_structure(self):
        input_data = {
            "workflow": {"key": "value"},
            "images": [{"name": "image1.png"}],  # Missing 'image' key
        }
        validated_data, error = handler.validate_input(input_data)
        self.assertIsNotNone(error)
        self.assertEqual(
            error, "'images' must be a list of objects with 'name' and 'image' keys"
        )

    def test_invalid_json_string_input(self):
        input_data = "invalid json"
        validated_data, error = handler.validate_input(input_data)
        self.assertIsNotNone(error)
        self.assertEqual(error, "Invalid JSON format in input")

    def test_valid_json_string_input(self):
        input_data = '{"workflow": {"key": "value"}}'
        validated_data, error = handler.validate_input(input_data)
        self.assertIsNone(error)
        self.assertEqual(validated_data["workflow"], {"key": "value"})

    def test_empty_input(self):
        input_data = None
        validated_data, error = handler.validate_input(input_data)
        self.assertIsNotNone(error)
        self.assertEqual(error, "Please provide input")


class TestServerAndQueue(unittest.TestCase):
    @patch("handler.requests.get")
    def test_check_server_server_up(self, mock_requests):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_requests.return_value = mock_response

        result = handler.check_server("http://127.0.0.1:8188", 1, 50)
        self.assertTrue(result)

    @patch("handler._is_comfyui_process_alive", return_value=None)
    @patch("handler.requests.get")
    def test_check_server_server_down(self, mock_requests, mock_alive):
        mock_requests.side_effect = handler.requests.RequestException()
        result = handler.check_server("http://127.0.0.1:8188", 1, 50)
        self.assertFalse(result)

    @patch("handler.requests.post")
    def test_queue_workflow(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"prompt_id": "123"}
        mock_post.return_value = mock_response

        result = handler.queue_workflow({"1": {"class_type": "X"}}, "client-1")
        self.assertEqual(result, {"prompt_id": "123"})

    @patch("handler.requests.get")
    def test_get_history(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"key": "value"}
        mock_get.return_value = mock_response

        result = handler.get_history("123")
        self.assertEqual(result, {"key": "value"})
        mock_get.assert_called_with("http://127.0.0.1:8188/history/123", timeout=30)


class TestUploadImages(unittest.TestCase):
    @patch("handler.requests.post")
    def test_upload_images_successful(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        test_image_data = base64.b64encode(b"Test Image Data").decode("utf-8")
        images = [{"name": "test_image.png", "image": test_image_data}]

        responses = handler.upload_images(images)
        self.assertEqual(responses["status"], "success")

    @patch("handler.requests.post")
    def test_upload_images_failed(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = handler.requests.RequestException(
            "400 Client Error"
        )
        mock_post.return_value = mock_response

        test_image_data = base64.b64encode(b"Test Image Data").decode("utf-8")
        images = [{"name": "test_image.png", "image": test_image_data}]

        responses = handler.upload_images(images)
        self.assertEqual(responses["status"], "error")


class TestGetAvailableModels(unittest.TestCase):
    @patch("handler.requests.get")
    def test_returns_all_model_types(self, mock_get):
        mock_get.return_value = _mock_object_info_response(_make_object_info())
        available = handler.get_available_models()
        self.assertEqual(
            available["checkpoints"],
            ["sd_xl_base_1.0.safetensors", "subdir/anime_v2.safetensors"],
        )
        self.assertEqual(available["loras"], ["detail_tweaker.safetensors"])
        self.assertEqual(available["vae"], ["sdxl_vae.safetensors"])
        self.assertEqual(
            available["text_encoders"],
            ["clip_l.safetensors", "t5xxl_fp16.safetensors"],
        )
        self.assertEqual(available["diffusion_models"], ["flux1-dev.safetensors"])
        self.assertEqual(available["upscale_models"], ["4x_ultrasharp.pth"])

    @patch("handler.requests.get")
    def test_returns_empty_dict_when_unreachable(self, mock_get):
        mock_get.side_effect = handler.requests.RequestException("boom")
        self.assertEqual(handler.get_available_models(), {})


class TestValidateWorkflowModels(unittest.TestCase):
    def _validate(self, workflow, object_info=None):
        with patch("handler.requests.get") as mock_get:
            mock_get.return_value = _mock_object_info_response(
                object_info if object_info is not None else _make_object_info()
            )
            return handler.validate_workflow_models(workflow)

    def test_all_references_present(self):
        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
            },
            "2": {
                "class_type": "LoraLoader",
                "inputs": {"lora_name": "detail_tweaker.safetensors", "model": ["1", 0]},
            },
            "3": {"class_type": "KSampler", "inputs": {"seed": 42}},
        }
        self.assertIsNone(self._validate(workflow))

    def test_subfolder_relative_name_present(self):
        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "subdir/anime_v2.safetensors"},
            }
        }
        self.assertIsNone(self._validate(workflow))

    def test_missing_checkpoint(self):
        workflow = {
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "does_not_exist.safetensors"},
            }
        }
        error = self._validate(workflow)
        self.assertIsNotNone(error)
        self.assertIn("does_not_exist.safetensors", error)
        self.assertIn("checkpoints", error)
        self.assertIn("/runpod-volume/models/checkpoints/", error)
        self.assertIn("sd_xl_base_1.0.safetensors", error)  # lists available

    def test_multiple_missing_models_reported_together(self):
        workflow = {
            "1": {
                "class_type": "LoraLoader",
                "inputs": {"lora_name": "missing_lora.safetensors"},
            },
            "2": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "missing_vae.safetensors"},
            },
            "3": {
                "class_type": "DualCLIPLoader",
                "inputs": {
                    "clip_name1": "missing_clip.safetensors",
                    "clip_name2": "t5xxl_fp16.safetensors",
                },
            },
        }
        error = self._validate(workflow)
        self.assertIsNotNone(error)
        self.assertIn("missing_lora.safetensors", error)
        self.assertIn("/runpod-volume/models/loras/", error)
        self.assertIn("missing_vae.safetensors", error)
        self.assertIn("/runpod-volume/models/vae/", error)
        self.assertIn("missing_clip.safetensors", error)
        self.assertIn("/runpod-volume/models/clip/", error)
        # the valid second clip must not be reported
        self.assertNotIn("'t5xxl_fp16.safetensors' not found", error)

    def test_list_placeholder_gets_specific_message(self):
        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "__list__"},
            }
        }
        error = self._validate(workflow)
        self.assertIsNotNone(error)
        self.assertIn("placeholder", error)
        self.assertIn("__list__", error)
        self.assertIn("UI default", error)

    def test_case_mismatch_gets_hint(self):
        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "SD_XL_Base_1.0.safetensors"},
            }
        }
        error = self._validate(workflow)
        self.assertIsNotNone(error)
        self.assertIn("case-sensitive", error)
        self.assertIn("sd_xl_base_1.0.safetensors", error)

    def test_linked_inputs_are_skipped(self):
        workflow = {
            "1": {
                "class_type": "LoraLoader",
                "inputs": {"lora_name": ["7", 0], "model": ["1", 0]},
            }
        }
        self.assertIsNone(self._validate(workflow))

    def test_unregistered_loader_is_skipped(self):
        # UnetLoaderGGUF is a known loader type but not present in this
        # ComfyUI build's /object_info — leave it to ComfyUI's validation.
        workflow = {
            "1": {
                "class_type": "UnetLoaderGGUF",
                "inputs": {"unet_name": "whatever.gguf"},
            }
        }
        self.assertIsNone(self._validate(workflow))

    def test_missing_input_image(self):
        workflow = {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": "/runpod-volume/not_uploaded.png"},
            }
        }
        error = self._validate(workflow)
        self.assertIsNotNone(error)
        self.assertIn("/runpod-volume/not_uploaded.png", error)
        self.assertIn("images", error)

    def test_annotated_image_reference_is_skipped(self):
        workflow = {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": "result.png [output]"},
            }
        }
        self.assertIsNone(self._validate(workflow))

    def test_fails_open_when_object_info_unreachable(self):
        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "does_not_exist.safetensors"},
            }
        }
        with patch("handler.requests.get") as mock_get:
            mock_get.side_effect = handler.requests.RequestException("network blip")
            self.assertIsNone(handler.validate_workflow_models(workflow))


class TestHandlerPreflightOrdering(unittest.TestCase):
    """The pre-flight must run before queue_workflow and not block valid jobs."""

    @patch("handler.queue_workflow")
    @patch("handler.check_server", return_value=True)
    @patch("handler.requests.get")
    def test_preflight_failure_short_circuits_queue(
        self, mock_get, mock_check_server, mock_queue
    ):
        mock_get.return_value = _mock_object_info_response(_make_object_info())
        job = {
            "id": "job-1",
            "input": {
                "workflow": {
                    "1": {
                        "class_type": "CheckpointLoaderSimple",
                        "inputs": {"ckpt_name": "missing.safetensors"},
                    }
                }
            },
        }
        result = handler.handler(job)
        self.assertIn("error", result)
        self.assertIn("missing.safetensors", result["error"])
        mock_queue.assert_not_called()

    @patch("handler.get_history")
    @patch("handler.queue_workflow")
    @patch("handler.websocket.WebSocket")
    @patch("handler.check_server", return_value=True)
    @patch("handler.requests.get")
    def test_valid_workflow_reaches_queue_unchanged(
        self, mock_get, mock_check_server, mock_ws_cls, mock_queue, mock_history
    ):
        mock_get.return_value = _mock_object_info_response(_make_object_info())
        mock_queue.return_value = {"prompt_id": "abc"}
        mock_history.return_value = {"abc": {"outputs": {"9": {"images": []}}}}

        mock_ws = MagicMock()
        mock_ws.recv.return_value = json.dumps(
            {"type": "executing", "data": {"node": None, "prompt_id": "abc"}}
        )
        mock_ws_cls.return_value = mock_ws

        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
            }
        }
        job = {"id": "job-2", "input": {"workflow": workflow}}
        result = handler.handler(job)

        self.assertNotIn("error", result)
        mock_queue.assert_called_once()
        self.assertEqual(mock_queue.call_args[0][0], workflow)


class TestHandlerAudioOutputs(unittest.TestCase):
    """SaveAudio* nodes report files under "audio"; they must be returned too."""

    @patch("handler.get_image_data", return_value=b"ID3fake-mp3")
    @patch("handler.get_history")
    @patch("handler.queue_workflow")
    @patch("handler.websocket.WebSocket")
    @patch("handler.check_server", return_value=True)
    @patch("handler.requests.get")
    def test_audio_output_returned_as_base64(
        self, mock_get, mock_check_server, mock_ws_cls, mock_queue, mock_history, mock_data
    ):
        mock_get.return_value = _mock_object_info_response(_make_object_info())
        mock_queue.return_value = {"prompt_id": "abc"}
        mock_history.return_value = {
            "abc": {
                "outputs": {
                    "107": {
                        "audio": [
                            {"filename": "song_00001_.mp3", "subfolder": "audio", "type": "output"}
                        ]
                    }
                }
            }
        }
        mock_ws = MagicMock()
        mock_ws.recv.return_value = json.dumps(
            {"type": "executing", "data": {"node": None, "prompt_id": "abc"}}
        )
        mock_ws_cls.return_value = mock_ws

        job = {"id": "job-3", "input": {"workflow": {"1": {"class_type": "Note", "inputs": {}}}}}
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BUCKET_ENDPOINT_URL", None)
            result = handler.handler(job)

        self.assertNotIn("error", result)
        self.assertEqual(len(result["images"]), 1)
        out = result["images"][0]
        self.assertEqual(out["filename"], "song_00001_.mp3")
        self.assertEqual(out["type"], "base64")
        self.assertEqual(base64.b64decode(out["data"]), b"ID3fake-mp3")
        mock_data.assert_called_once_with("song_00001_.mp3", "audio", "output")


if __name__ == "__main__":
    unittest.main()


class TestHandlerTts(unittest.TestCase):
    """{"tts": {...}} jobs go to the TTS module and never touch ComfyUI."""

    @patch("handler.check_server")
    def test_tts_job_dispatched(self, mock_check_server):
        import tts
        with patch.object(tts, "run", return_value={"images": [{"filename": "voice.wav"}]}) as run:
            result = handler.handler({"id": "j1", "input": {"tts": {"text": "hi", "ref_audio": "AAAA"}}})
        run.assert_called_once_with({"text": "hi", "ref_audio": "AAAA"})
        mock_check_server.assert_not_called()
        self.assertEqual(result["images"][0]["filename"], "voice.wav")

    def test_tts_validates_input(self):
        import tts
        self.assertIn("error", tts.run({"text": ""}))
        self.assertIn("ref_audio", tts.run({"text": "hello"})["error"])
