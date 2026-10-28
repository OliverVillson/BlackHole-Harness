"""Write install.sh's banner from the TUI art, so the installer shows the same hole and
wordmark as the TUI instead of a fifth drawing. Rerun after any change to
blackhole_tui/art.py or theme.py:

    python harness/scripts/install_banner.py

It rewrites the text between the BANNER markers in install.sh and in its copy,
blackhole/cloud/install.sh. install.sh must stay plain bash (it runs before Python
is set up), so the art goes in as 256-colour escape sequences, written as \\033 and
expanded by printf %b. Only full and half blocks are used for the hole: shade
characters (light/medium/dark shade) render as a dither in many terminal fonts.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "harness"))

from rich.console import Console  # noqa: E402
from rich.text import Text  # noqa: E402

from blackhole_tui import art  # noqa: E402

TARGETS = [ROOT / "install.sh", ROOT / "blackhole" / "cloud" / "install.sh"]
START, END = "# >>> BANNER (harness/scripts/install_banner.py)", "# <<< BANNER"
WIDTH = 74  # the wordmark; the banner needs a terminal at least this wide


def _ansi(text: Text, color: bool) -> str:
    buf = io.StringIO()
    Console(file=buf, width=200, color_system="256" if color else None,
            force_terminal=color, legacy_windows=False).print(text, end="")
    return buf.getvalue().replace("\033", "\\033").rstrip()


def colour() -> str:
    hole = art.render(40, 10)
    pad = " " * ((WIDTH - 40) // 2)
    t = Text(no_wrap=True)
    for line in hole.split("\n"):
        t.append(pad)
        t.append_text(line)
        t.append("\n")
    t.append_text(art.banner(tag=""))
    return _ansi(t, True)


def mono() -> str:
    return _ansi(art.banner(tag=""), False)


def block() -> str:
    return "\n".join([
        START,
        "BANNER_COLOUR=$(cat <<'EOF'",
        colour(),
        "EOF",
        ")",
        "BANNER_MONO=$(cat <<'EOF'",
        mono(),
        "EOF",
        ")",
        END,
    ])


def splice(src: str) -> str:
    a, b = src.index(START), src.index(END) + len(END)
    return src[:a] + block() + src[b:]


if __name__ == "__main__":
    for p in TARGETS:
        p.write_text(splice(p.read_text()))
        print("wrote", p.relative_to(ROOT))
