# blackholeharness.dev

Static landing page. No build step: deploy this folder as-is.

Vercel (`vercel.json` here): import the repo, framework preset **Other**, root directory `landing`, build command empty, output directory `.`. `.vercelignore` keeps the sources (`page.html`, `build.py`) off the site. Point blackholeharness.dev at the project in Vercel's Domains settings.

- `index.html` the site (inline CSS + one small script that draws the black hole and runs the copy buttons)
- `page.html` the same page without the document wrapper, used for the claude.ai preview; edit this one, then regenerate `index.html` with `python3 build.py`
- `shots/` TUI screenshots (WebP, from blackhole-tui/shots)
- `favicon.svg`
