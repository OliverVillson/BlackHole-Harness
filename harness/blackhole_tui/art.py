"""Logo art. The black hole is drawn, not traced: a small ray-free model of a
Gargantua-style hole (shadow, photon ring, a tilted accretion disk in front and
its lensed far side bent over the top), sampled onto half-block pixels so each
terminal cell holds two square-ish pixels. Brightness is snapped to a six-step
black-hole palette, so the terminal's own black is the event horizon.

The art is deterministic; `python -m blackhole_tui.art` prints both sizes.
"""

from __future__ import annotations

import math

from rich.text import Text

from blackhole_tui.theme import BANNER_GRADIENT, HEAT, SHADOW_EDGE

# "BLACKHOLE" in figlet ansi_shadow. Each letter takes one step of the heat
# gradient, left to right; the box-drawing shadow strokes stay dark red.
BANNER = """\
██████╗ ██╗      █████╗  ██████╗██╗  ██╗██╗  ██╗ ██████╗ ██╗     ███████╗
██╔══██╗██║     ██╔══██╗██╔════╝██║ ██╔╝██║  ██║██╔═══██╗██║     ██╔════╝
██████╔╝██║     ███████║██║     █████╔╝ ███████║██║   ██║██║     █████╗
██╔══██╗██║     ██╔══██║██║     ██╔═██╗ ██╔══██║██║   ██║██║     ██╔══╝
██████╔╝███████╗██║  ██║╚██████╗██║  ██╗██║  ██║╚██████╔╝███████╗███████╗
╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚══════╝"""


def _letter_of_column() -> list[int]:
    """Letter index (0..8) for every column of BANNER: the letters are the runs of
    columns that hold a solid block in any row."""
    lines = BANNER.splitlines()
    width = max(len(l) for l in lines)
    solid = [any(x < len(l) and l[x] == "█" for l in lines) for x in range(width)]
    letter, out = -1, []
    for x in range(width):
        if solid[x] and (x == 0 or not solid[x - 1]):
            letter += 1
        out.append(max(letter, 0))
    return out


def banner(tag: str = "harness") -> Text:
    lines = BANNER.splitlines()
    letters = _letter_of_column()
    n = letters[-1] + 1
    out = Text(no_wrap=True)
    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            if ch == "█":
                out.append(ch, style=BANNER_GRADIENT[letters[x] * len(BANNER_GRADIENT) // n])
            elif ch != " ":
                out.append(ch, style=SHADOW_EDGE)
            else:
                out.append(" ")
        if y == len(lines) - 2 and tag:
            out.append("  " + tag, style=f"bold {HEAT[-1]}")
        out.append("\n")
    out.rstrip()
    return out


DISK_IN, DISK_OUT = 1.4, 3.1
TILT = 0.17  # sin of the disk's inclination: how flat it looks


def _intensity(x: float, y: float, px: float) -> float:
    """Brightness 0..1 at (x, y), in units of the shadow radius; +y is down.
    px is the size of one pixel in the same units, so the photon ring can be
    drawn one pixel wide at any size instead of breaking into dots."""
    r = math.hypot(x, y)

    # the near side of the disk passes in front of the shadow
    if y >= 0:
        rho = math.hypot(x, y / TILT)
        if DISK_IN <= rho <= DISK_OUT:
            phi = math.atan2(y / TILT, x)
            fall = (DISK_OUT - rho) / (DISK_OUT - DISK_IN)
            beam = 1.0 + 0.45 * (-x / rho)  # approaching side (left) is brighter
            swirl = 0.9 + 0.1 * math.sin(phi * 3 + rho * 2)  # a few broad arcs, no speckle
            return min((0.25 + 0.75 * fall ** 1.4) * beam * swirl, 1.0)

    if r < 1.0:
        return 0.0  # event horizon

    # photon ring hugging the shadow: every pixel within 0.6 px of its centre line is
    # fully lit, which gives an unbroken one-pixel ring at any size, then a short glow
    d = abs(r - 1.0 - 0.5 * px) / px
    ring = 1.0 if d <= 0.6 else math.exp(-((d - 0.6) / 0.7) ** 2)

    # far side of the disk, lensed into a halo: thick over the top and right down
    # to the disk, thin underneath
    above = min(1.0, max(0.0, 1.0 - y / 0.45))
    halo = 0.0
    edge = 1.57 + 0.5 * above
    if r < edge:
        t = (r - 1.0) / (edge - 1.0)
        halo = (1 - t) ** 1.6 * (0.5 + 0.35 * above)
        halo *= 1.0 + 0.35 * (-x / r)
        halo *= 0.94 + 0.06 * math.sin(math.atan2(y, x) * 2 + r * 2)

    # far side of the disk, behind the hole, visible past the shadow's edge
    back = 0.0
    if y < 0:
        rho = math.hypot(x, y / TILT)
        if DISK_IN <= rho <= DISK_OUT:
            back = 0.55 * (DISK_OUT - rho) / (DISK_OUT - DISK_IN) * (1.0 + 0.4 * (-x / rho))

    # faint glow around everything
    glow = 0.10 * math.exp(-max(0.0, r - 1.1) * 1.6) if r < DISK_OUT + 0.3 else 0.0
    return min(1.0, max(ring * 0.95, halo, back, glow))


def _palette_index(v: float) -> int:
    # cut points tuned so the disk shows every step from ember to white-hot
    cuts = (0.06, 0.16, 0.30, 0.46, 0.64, 0.82)
    for i, c in enumerate(cuts):
        if v < c:
            return i  # 0 is "not lit": terminal black
    return len(cuts)


def render(cols: int, rows: int, scale: float = 1.0) -> Text:
    """The hole in cols cells wide and at most rows tall, two pixels per cell
    (half blocks). Unlit rows at the top and bottom are dropped."""
    px_h = rows * 2
    # fit the disk (radius DISK_OUT) to the width, keeping pixels square
    unit = (cols / 2) / ((DISK_OUT + 0.1) * scale)
    grid = []
    for py in range(px_h):
        line = []
        for px in range(cols):
            x = (px + 0.5 - cols / 2) / unit
            y = (py + 0.5 - px_h / 2) / unit
            line.append(_palette_index(_intensity(x, y, 1 / unit)))
        grid.append(line)

    lit = [i for i, l in enumerate(grid) if any(l)]
    grid = grid[lit[0] // 2 * 2: (lit[-1] // 2 + 1) * 2] if lit else grid
    rows = len(grid) // 2
    palette = [None, *HEAT]  # 0 = unlit
    out = Text(no_wrap=True)
    for cy in range(rows):
        top, bot = grid[2 * cy], grid[2 * cy + 1]
        for t, b in zip(top, bot):
            ct, cb = palette[t], palette[b]
            if ct is None and cb is None:
                out.append(" ")
            elif ct == cb:
                out.append("█", style=ct)
            elif cb is None:
                out.append("▀", style=ct)
            elif ct is None:
                out.append("▄", style=cb)
            else:
                out.append("▀", style=f"{ct} on {cb}")
        if cy < rows - 1:
            out.append("\n")
    return out


# 50 x 18: the default left column, as REAPer's small reaper
HOLE = render(50, 18)
# 64 x 26: when the terminal is at least 150 x 66
HOLE_LARGE = render(64, 26)


if __name__ == "__main__":
    from rich.console import Console

    c = Console()
    c.print(banner())
    c.print(HOLE)
    c.print(HOLE_LARGE)
