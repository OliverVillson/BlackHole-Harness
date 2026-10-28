# Every layer to 4 bits.
# int3, int8, fp8, mlx4, nvfp4, hex3 and hex4 work here too.

from blackhole_dev.recipe import Recipe

recipe = Recipe("me/int4", "1.0.0", title="Everything to 4 bits")

recipe.fn("quant", select="layers[*]", format="int4")
