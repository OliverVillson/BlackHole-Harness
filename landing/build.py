"""Wrap page.html (the claude.ai preview source) into a full index.html."""
from pathlib import Path

here = Path(__file__).parent
p = (here / "page.html").read_text()
head, rest = p.split("<style>", 1)
style, body = rest.split("</style>", 1)
(here / "index.html").write_text(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
{head.strip()}
<meta property="og:title" content="BlackHole-Harness">
<meta property="og:description" content="Shrink any open-weight model until it runs on your own PC.">
<meta property="og:image" content="https://blackholeharness.dev/shots/05-running.webp">
<meta name="theme-color" content="#000000">
<link rel="icon" href="favicon.svg" type="image/svg+xml">
<style>{style}</style>
</head>
<body>
{body.strip()}
</body>
</html>
""")
