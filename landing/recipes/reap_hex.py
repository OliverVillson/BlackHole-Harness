# REAP, then REAQ gives each expert 8 or 16 levels and
# HEX quantizes them onto learned tables. Packed HEX out.

from blackhole_dev.recipe import Recipe, gate

recipe = Recipe("me/reap-hex", "1.0.0",
                title="REAP, then HEX codebooks",
                targets={"architectures": ["moe"]})

calib = recipe.input("calib", "path", default="my-prompts.txt")

recipe.fn("reap-score", calib=calib)
recipe.fn("reap-prune", ratio=0.25)
recipe.edit([{"fn": "reaq", "avg_bits": 3.5},
             {"fn": "hex-quantize", "calib": calib}],
            gates=[gate("hex-quantize.rel_mse", "<=", 0.2)],
            pack="hex")
