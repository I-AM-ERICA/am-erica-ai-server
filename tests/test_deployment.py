from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("manage", ROOT / "scripts/manage.py")
manage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manage)
spec = importlib.util.spec_from_file_location("serve", ROOT / "runtime/serve.py")
serve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve)


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        shutil.copytree(ROOT / "deploy", self.root / "deploy")

    def tearDown(self):
        self.temp.cleanup()

    def test_init_keeps_existing_keys_and_uses_host_paths(self):
        manage.init(self.root)
        before = (self.root / ".env.gpu").read_bytes()
        manage.init(self.root)
        self.assertEqual(before, (self.root / ".env.gpu").read_bytes())
        gpu, cpu = (manage.read_env(self.root / f".env.{g}") for g in manage.GROUPS)
        self.assertEqual(gpu["AI_INTERNAL_KEY"], cpu["AI_INTERNAL_KEY"])
        self.assertEqual(64, len(gpu["AI_INTERNAL_KEY"]))
        self.assertTrue(gpu["POLICY_MODEL_DIR"].startswith(str(self.root.resolve())))
        self.assertEqual(0o600, (self.root / ".env.gpu").stat().st_mode & 0o777)

    def test_partial_init_reuses_existing_key(self):
        manage.init(self.root)
        first = manage.read_env(self.root / ".env.gpu")["AI_INTERNAL_KEY"]
        (self.root / ".env.cpu").unlink()
        manage.init(self.root)
        self.assertEqual(first, manage.read_env(self.root / ".env.cpu")["AI_INTERNAL_KEY"])

    def test_literal_env_values_never_execute_or_expand(self):
        path = self.root / "config"
        path.write_text("KEY='$(touch /tmp/should-never-run) # literal'\n")
        self.assertEqual("$(touch /tmp/should-never-run) # literal", manage.read_env(path)["KEY"])
        for text in ("KEY=$HOME\n", "KEY=one\nKEY=two\n", "export KEY=value\n"):
            path.write_text(text)
            with self.assertRaises(manage.ConfigError):
                manage.read_env(path)

    def test_validation_reports_missing_keys_models_and_store_corpus(self):
        manage.init(self.root)
        cfg = manage.read_env(self.root / ".env.gpu")
        errors = "\n".join(manage.validate("gpu", cfg))
        for expected in ("SERPAPI_API_KEY", "POLICY_MODEL_DIR", "CUSTOMER_ADAPTER_DIR", "real store docs"):
            self.assertIn(expected, errors)
        self.assertNotIn(cfg["AI_INTERNAL_KEY"], errors)

    def test_valid_cpu_config_and_invalid_port(self):
        manage.init(self.root)
        cfg = manage.read_env(self.root / ".env.cpu")
        directory = Path(cfg["CHRONOS_MODEL_DIR"])
        directory.mkdir(parents=True)
        for name in ("config.json", "generation_config.json", "model.safetensors"):
            (directory / name).write_text("fixture")
        cfg["CLOVA_STUDIO_API_KEY"] = "test-value"
        self.assertEqual([], manage.validate("cpu", cfg))
        cfg["PROMOTION_PORT"] = cfg["MANAGEMENT_PORT"]
        self.assertIn("Service ports must be distinct", manage.validate("cpu", cfg))
        cfg["POSTER_PUBLIC_BASE_URL"] = "http://["
        self.assertTrue(any("POSTER_PUBLIC_BASE_URL" in error for error in manage.validate("cpu", cfg)))

    def test_backend_config_disables_local_processes_and_customer_key_file(self):
        manage.init(self.root)
        settings = manage.configs(manage.GROUPS, self.root)
        output = manage.backend_env(settings, "10.0.1.20", "10.0.2.20")
        for expected in ("AI_POLICY_HOST=10.0.1.20", "AI_PROMOTION_HOST=10.0.2.20", "AI_MOCK_ENABLED=false", "AI_CUSTOMER_KEY_FILE=\n", "AI_PROMOTION_AUTOSTART=false", "AI_MANAGEMENT_AUTOSTART=false"):
            self.assertIn(expected, output)
        settings["cpu"]["AI_INTERNAL_KEY"] = "different"
        with self.assertRaises(manage.ConfigError):
            manage.backend_env(settings, "127.0.0.1", "127.0.0.1")

    def test_compose_uses_validated_env_over_parent_shell_and_never_shell_executes(self):
        manage.init(self.root)
        cfg = manage.read_env(self.root / ".env.cpu")
        cfg["CLOVA_STUDIO_API_KEY"] = "literal-$-value"
        with patch.dict(os.environ, {"PROMOTION_PORT": "9999"}), patch.object(manage.subprocess, "run") as run:
            manage.compose("cpu", cfg, ["up", "--detach"], self.root)
        kwargs = run.call_args.kwargs
        self.assertEqual(cfg["PROMOTION_PORT"], kwargs["env"]["PROMOTION_PORT"])
        self.assertEqual("literal-$-value", kwargs["env"]["CLOVA_STUDIO_API_KEY"])
        self.assertNotIn("shell", kwargs)

    def test_gpu_customer_config_removes_cpu_quantization(self):
        aliases = self.root / "aliases.json"
        aliases.write_text('{"1":"store-a"}')
        with patch.dict(os.environ, {"CUSTOMER_DEVICE": "cuda", "CUSTOMER_STORE_ALIASES": str(aliases)}, clear=True):
            cfg = serve.customer_config()
        self.assertEqual("cuda", cfg["base"]["device"])
        self.assertIsNone(cfg["base"]["cpu_quantization"])
        self.assertEqual({"1": "store-a"}, cfg["store_aliases"])
        self.assertEqual("0.0.0.0", cfg["host"])

    def test_service_adapters_fail_closed_and_use_original_entry_points(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                serve.command("policy")
        with patch.dict(os.environ, {"AI_INTERNAL_KEY": "test", "SERPAPI_API_KEY": "test"}, clear=True):
            module, args = serve.command("policy")
            self.assertEqual("scripts.serve_policy", module)
            self.assertIn("mvp", args)
            self.assertIn("0.0.0.0", args)
            module, _ = serve.command("management")
            self.assertEqual("chronos_sales.management_grpc", module)
            self.assertEqual("", os.environ["CUDA_VISIBLE_DEVICES"])

    def test_private_config_writer_refuses_overwrite_and_symlinks(self):
        path = self.root / "config"
        manage.write_private(path, "first")
        with self.assertRaises(FileExistsError):
            manage.write_private(path, "second")
        link = self.root / "link"
        link.symlink_to(path)
        with self.assertRaises(OSError):
            manage.write_private(link, "third", force=True)
        self.assertEqual("first", path.read_text())

    def test_model_import_omits_training_state_and_preserves_existing_artifacts(self):
        source = self.root / "originals"
        selection = json.loads((ROOT / "services/policy/upstream/serving/model-selection.json").read_text())
        policy = source / "Search-R1/outputs/audit-input" / Path(selection["checkpoint"]).relative_to("outputs")
        for base, names in (
            (policy, ("config.json", "metadata.json", "actor/adapter_model.safetensors", "actor/adapter_config.json", "tokenizer/tokenizer.json", "optimizer.pt")),
            (source / "Scrable/adapter", ("adapter_config.json", "adapter_model.safetensors")),
            (source / "Chronos/models", ("config.json", "generation_config.json", "model.safetensors")),
        ):
            for name in names:
                path = base / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("model fixture")
        target_selection = self.root / "services/policy/upstream/serving/model-selection.json"
        target_selection.parent.mkdir(parents=True)
        target_selection.write_text(json.dumps(selection))
        manage.import_models(source, self.root)
        self.assertFalse((self.root / "data/models/policy/optimizer.pt").exists())
        target = self.root / "data/models/customer/adapter_model.safetensors"
        target.write_text("keep this model")
        manage.import_models(source, self.root)
        self.assertEqual("keep this model", target.read_text())

    def test_original_model_and_contract_snapshots_are_unmodified(self):
        self.assertEqual([], manage.verify_sources(ROOT))


if __name__ == "__main__":
    unittest.main()
