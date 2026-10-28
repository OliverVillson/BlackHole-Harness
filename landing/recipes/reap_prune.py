# REAP: score every expert on your own text,
# then drop the quarter you never use.

from blackhole_dev.recipe import Recipe, gate

recipe = Recipe("me/reap-25", "1.0.0",
                title="Drop the experts I never use",
                targets={"architectures": ["moe"]})

calib = recipe.input("calib", "path", default="my-prompts.txt",
                     help="Text like what you will ask it")

recipe.fn("reap-score", calib=calib)
recipe.fn("reap-prune", ratio=0.25,
          gates=[gate("reap-prune.experts_dropped", ">", 0)])
