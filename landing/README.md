# blackholeharness.dev

Static landing page. No build step: deploy this folder as-is.

Vercel (`vercel.json` here): import the repo, framework preset **Other**, root directory `landing`, build command empty, output directory `.`. `.vercelignore` keeps the sources (`page.html`, `build.py`) off the site. Point blackholeharness.dev at the project in Vercel's Domains settings.

- `index.html` the site (inline CSS + one small script that draws the black hole and runs the copy buttons)
- `page.html` the same page without the document wrapper, used for the claude.ai preview; edit this one, then run `python3 build.py`: it fills the recipe blocks from `recipes/*.py` (syntax-highlighted) and regenerates `index.html`
- `recipes/` the recipe files the page shows, also served as downloads
- `shots/` TUI screenshots (WebP, from blackhole-tui/shots), BlackHole-Dev screenshots (`dev-*.webp`) and theme thumbnails (`theme-*.webp`)
- `favicon.svg`
