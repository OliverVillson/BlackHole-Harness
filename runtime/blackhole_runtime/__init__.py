"""GPU kernels for BlackHole-Harness formats outside MLX.

hex_ref.py   the torch reference for HEX experts (blackhole.anatomy.hex is the source of truth)
hex_moe.py   Triton fused MoE over HEX experts: grouped GEMM over the chosen experts that reads
             the per-expert level tables, fp8 block scales and 3/4-bit indices directly
hex_check.py numerical check against the reference + micro-benchmark against vLLM's int4 MoE
vllm_hex.py  the "blackhole-hex" quantization method for vLLM (loaded as a vLLM plugin)
convert.py   a BlackHole MLX HEX export -> a "blackhole-hex" folder vLLM can serve
"""
