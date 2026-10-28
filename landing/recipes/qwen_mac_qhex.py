# REAP/Q-HEX Mac coder (Qwen3.6): the recipe behind the 9.8 GB build. Every step is a built-in
# function you can open and fork in BlackHole-Dev.

from blackhole_dev.recipe import Recipe


recipe = Recipe(
    "blackhole/qwen-mac-qhex",
    "1.0.0",
    title="REAP/Q-HEX Mac coder (Qwen3.6)",
    description=(
        "Qwen3.6-35B-A3B for a 16 GB Apple-silicon Mac: REAP drops 31.25% of routed experts on code "
        "(256 -> 176 per layer); distillation on the teacher's shortest verified answers plus two "
        "on-policy rounds; MLX export with HEX experts (REAQ tiers: top 10% 16-level, rest 8-level) "
        "and 3-bit GPTQ attention; then qheal around the quantized model. 9.8 GB, KL 0.070."
    ),
    author="BlackHole-Harness",
    tags=["mac", "code", "moe", "lab", "reaq", "distill", "qwen"],
    meta={"quant": "mlx-3bit", "preset": "qwen-mac-qhex"},
)

recipe.fn("calibrate", calib="code")
# drop 31.25% of routed experts, calibrated on code
recipe.fn("reap-gpu", ratio=0.3125)
# shortest verified traces + on-policy distillation, layer and router matching
recipe.fn("distil",
          calib="contest",
          samples=3,
          prompts=768,
          max_tokens=16384,
          seq=18432,
          expert_rank=16,
          layer_weight=1.0,
          router_weight=0.5,
          minutes=150,
          opd_rounds=2,
          opd_prompts=256,
          opd_minutes=90)
# MLX: HEX experts, REAQ tiers (top 10% 16-level, rest 8-level), GPTQ attention q/o 3-bit, full output vocab
recipe.fn("mlx-export",
          hex=1,
          bits=3,
          group=128,
          hot=0.1,
          hot_bits=4,
          attn_gptq=1,
          qo_bits=3,
          head_bits=4,
          embed_bits=3,
          head_vocab=0)
# distil again around the quantized model
recipe.fn("qheal", minutes=45)
recipe.fn("bench", models="original,mlx")
recipe.fn("package")
