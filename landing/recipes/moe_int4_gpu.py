# The MoE default on a GPU box: REAP 25% + GPTQ int4,
# benchmarked against the original and packaged.

from blackhole_dev.recipe import Recipe

recipe = Recipe("me/moe-int4", "1.0.0",
                title="REAP 25% + GPTQ int4")

recipe.fn("calibrate", calib="code")
recipe.fn("reap-gpu", ratio=0.25)
recipe.fn("gptq")
recipe.fn("bench")
recipe.fn("package")
