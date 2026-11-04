# REAP: score every expert on sample text,
# then drop the least used quarter.

from blackhole_dev.recipe import Recipe, gate

recipe = Recipe("examples/reap-25", "1.0.0",
                title="Drop the least used experts",
                targets={"architectures": ["moe"]})

calib = recipe.input("calib", "path", default="prompts.txt",
                     help="Text like the prompts the model will get")

recipe.fn("reap-score", calib=calib)
recipe.fn("reap-prune", ratio=0.25,
          gates=[gate("reap-prune.experts_dropped", ">", 0)])
