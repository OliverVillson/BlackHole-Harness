"""install.sh's banner is generated from the TUI art; it must not drift from it."""
import importlib.util
import re
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "install_banner.py"
spec = importlib.util.spec_from_file_location("install_banner", SCRIPT)
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)


def test_install_sh_banner_matches_tui_art():
    for p in gen.TARGETS:
        assert gen.block() in p.read_text(), f"python {SCRIPT.relative_to(gen.ROOT)}"


def test_banner_fits_80_columns_and_avoids_shade_glyphs():
    for text in (gen.colour(), gen.mono()):
        plain = re.sub(r"\\033\[[0-9;]*m", "", text)
        assert max(len(l) for l in plain.splitlines()) <= gen.WIDTH <= 80
        assert not set("░▒▓") & set(plain)
