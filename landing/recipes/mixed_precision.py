# Your own step: a function defined right in the recipe
# file, shipped inside the bundle with it.

from blackhole_dev import function
from blackhole_dev.recipe import Recipe

recipe = Recipe("me/mixed", "1.0.0",
                title="8-bit attention, 4-bit experts")


@function("mixed-precision", per="model")
def mixed(m, attn: str = "int8", experts: str = "int4"):
    m["layers[*].attn"].quant(attn)
    m["experts"].quant(experts)
    return {"experts": len(m["experts"])}


recipe.fn("mixed-precision", attn="int8", experts="int4")
