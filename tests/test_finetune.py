from pathlib import Path

from router.finetune import lora_command


def test_lora_command_masks_prompt_and_targets_adapter(tmp_path: Path):
    cmd = lora_command("some/model", tmp_path / "mlx", tmp_path / "adapters" / "1.5b", iters=10)
    assert cmd[1:4] == ["-m", "mlx_lm", "lora"]
    assert "--mask-prompt" in cmd
    assert "--train" in cmd
    assert cmd[cmd.index("--iters") + 1] == "10"
    assert cmd[cmd.index("--adapter-path") + 1].endswith("1.5b")
