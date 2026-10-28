"""Fill the recipe blocks in page.html from recipes/*.py, then wrap page.html (the claude.ai
preview source) into a full index.html."""
import html
import io
import keyword
import re
import tokenize
from pathlib import Path

here = Path(__file__).parent


def highlight(src: str) -> str:
    """Python source as escaped HTML with spans: k keyword, s string, c comment, f call or decorator, n number."""
    lines = src.splitlines(keepends=True)
    offs = [0]
    for ln in lines:
        offs.append(offs[-1] + len(ln))
    toks = [t for t in tokenize.generate_tokens(io.StringIO(src).readline)]
    out, pos = [], 0
    for i, t in enumerate(toks):
        if t.type in (tokenize.ENDMARKER,):
            break
        a = offs[t.start[0] - 1] + t.start[1]
        b = offs[t.end[0] - 1] + t.end[1]
        if a < pos:
            continue
        out.append(html.escape(src[pos:a], quote=False))
        text = html.escape(src[a:b], quote=False)
        nxt = next((u for u in toks[i + 1:] if u.type not in (tokenize.NL, tokenize.NEWLINE)), None)
        prev = toks[i - 1] if i else None
        cls = None
        if t.type == tokenize.COMMENT:
            cls = "c"
        elif t.type == tokenize.STRING or tokenize.tok_name[t.type].startswith("FSTRING"):
            cls = "s"
        elif t.type == tokenize.NUMBER:
            cls = "n"
        elif t.type == tokenize.NAME and (keyword.iskeyword(t.string) or t.string in ("None", "True", "False")):
            cls = "k"
        elif t.type == tokenize.NAME and ((nxt and nxt.string == "(") or (prev and prev.string == "@")):
            cls = "f"
        out.append(f'<span class="{cls}">{text}</span>' if cls else text)
        pos = b
    out.append(html.escape(src[pos:], quote=False))
    return "".join(out).rstrip("\n")


page = (here / "page.html").read_text()
page = re.sub(r"(<!--recipe:([\w.]+)-->).*?(<!--/recipe-->)",
              lambda m: m[1] + highlight((here / "recipes" / m[2]).read_text()) + m[3], page, flags=re.S)
(here / "page.html").write_text(page)

head, rest = page.split("<style>", 1)
style, body = rest.split("</style>", 1)
(here / "index.html").write_text(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
{head.strip()}
<meta property="og:title" content="BlackHole-Dev">
<meta property="og:description" content="Write a recipe, collapse an open model until it fits your machine.">
<meta property="og:image" content="https://blackholeharness.dev/shots/dev-hero.webp">
<meta name="theme-color" content="#000000">
<link rel="icon" href="favicon.svg" type="image/svg+xml">
<style>{style}</style>
</head>
<body>
{body.strip()}
</body>
</html>
""")
