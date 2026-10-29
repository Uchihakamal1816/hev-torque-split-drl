# Real drive cycles

| File | Source | Length | Use |
|---|---|---|---|
| `raw/uddscol.txt` | US EPA, Urban Dynamometer Driving Schedule (`https://www.epa.gov/sites/default/files/2015-10/uddscol.txt`) | 1370 s | training |
| `raw/hwycol.txt` | US EPA, Highway Fuel Economy Test (`https://www.epa.gov/sites/default/files/2015-10/hwycol.txt`) | 766 s | training |
| `raw/V_class3a.txt` | WLTC class 3a speed trace, from the JRC `wltp` package (`https://github.com/ankostis/wltp/raw/master/wltp/cycles/V_class3a.txt`) | 1800 s | held-out test |
| `raw/V_class3b.txt` | WLTC class 3b speed trace, same repo (`.../V_class3b.txt`) | 1800 s | held-out test |

`cycles/*.csv` are the converted speed traces in m/s, made by `scripts/prepare_real_cycles.py`.
Speeds: EPA files are in mph, WLTC files in km/h; both are converted.

The WLTC files come from a third-party repository. Check its licence before redistributing
these files outside this project.
