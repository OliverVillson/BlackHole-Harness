"""The TUI: `blackhole` (alias `bh`) over SSH on the GPU VM. Owned by the "Black hole TUI"
thread; its code (jobs.py, theme.py) is in
/mnt/project-files/blackhole-tui/blackhole_tui/ and moves here as is. Architecture carried over from REAPer's reaper_tui: a Textual app that starts a
job detached (blackhole.core.runner.start_detached), tails <job>/events.log, reattaches
to the newest job on start, slash commands plus F-keys, and a marketplace table.

Screens: Home (logo, machine, current job), Setup (schema inputs: simple mode shows
required non-advanced ones, advanced shows all and the stage list), Market (schemas,
filters, "Fits your PC"), Describe (frontier-model schema builder chat), Run (stage list
with progress bars from events, gates, metrics), Plan (hardware + per-stage plan + cost).
"""

# The black-hole palette: the TUI, banners and the docs use these and nothing else.
PALETTE = {
    "void": "#000000",        # background: the event horizon
    "ember": "#ff6a00",       # primary: the accretion disk's hot orange
    "flare": "#ffae42",       # highlights, progress fill
    "redshift": "#d7261e",    # errors, failed gates
    "crimson": "#9e1b07",     # frames
    "ash": "#4a0d02",         # quiet borders, empty bar cells
    "dim": "#8a5a44",         # secondary text
    "text": "#f0dccb",        # body text
    "photon": "#f5efe6",      # the bit of white: titles, the photon ring
}
