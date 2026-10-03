"""The architecture diagram: training, promotion, monitoring and serving.

The README used to carry this as an ASCII box diagram, which is honest about
the shape but hard to read and does not survive copying out of a terminal.
This draws the same topology as an SVG, in a light and a dark palette so it
does not show up as a white rectangle on GitHub's dark theme -- the README
picks between the two with a theme-conditional <picture> element.

This diagram describes structure, not a specific run: unlike the figures in
some sibling repositories, there is no pinned results.json here to read
numbers from, because this platform is meant to be run, not replayed from a
frozen artifact. The MAE and drift figures quoted in the README come from one
real run of `mlplatform train` / `mlplatform monitor`, recorded in prose.

    python scripts/figures/generate_diagram.py

Output: docs/architecture.svg, docs/architecture-dark.svg
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
OUT = REPO / "docs"

FONT = "Segoe UI, -apple-system, Helvetica, Arial, sans-serif"

PALETTES = {
    "light": {
        "ink": "#111827",
        "muted": "#6B7280",
        "faint": "#9CA3AF",
        "hair": "#E5E7EB",
        "blue": "#2563EB",
        "blue_bg": "#EFF6FF",
        "green": "#059669",
        "green_bg": "#ECFDF5",
        "amber": "#D97706",
        "amber_bg": "#FFFBEB",
        "grey_bg": "#F9FAFB",
    },
    "dark": {
        "ink": "#E6EDF3",
        "muted": "#8B949E",
        "faint": "#6E7681",
        "hair": "#30363D",
        "blue": "#58A6FF",
        "blue_bg": "#0D2847",
        "green": "#3FB950",
        "green_bg": "#0D2818",
        "amber": "#D29922",
        "amber_bg": "#2D2410",
        "grey_bg": "#161B22",
    },
}


def header(w, h, title, subtitle, p):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}" font-family="{FONT}">\n'
        f"  <defs><style>\n"
        f"    .h {{ fill:{p['ink']}; font-size:19px; font-weight:600; }}\n"
        f"    .s {{ fill:{p['muted']}; font-size:12.5px; }}\n"
        f"    .lbl {{ fill:{p['ink']}; font-size:13.5px; font-weight:600; }}\n"
        f"    .sub {{ fill:{p['muted']}; font-size:11px; }}\n"
        f"    .cap {{ fill:{p['faint']}; font-size:11.5px; }}\n"
        f"    .lane {{ fill:{p['faint']}; font-size:11.5px; font-style:italic; }}\n"
        f"  </style></defs>\n"
        f'  <text class="h" x="30" y="36">{title}</text>\n'
        f'  <text class="s" x="30" y="58">{subtitle}</text>\n'
    )


def box(x, y, w, h, fill, edge, title, lines):
    s = (
        f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" '
        f'stroke="{edge}" stroke-width="1.4"/>\n'
        f'  <text class="lbl" x="{x + w / 2}" y="{y + h / 2 - 2 if not lines else y + 25}" '
        f'text-anchor="middle">{title}</text>\n'
    )
    for i, ln in enumerate(lines):
        s += (
            f'  <text class="sub" x="{x + w / 2}" y="{y + 44 + i * 15}" '
            f'text-anchor="middle">{ln}</text>\n'
        )
    return s


def arrow(x1, y1, x2, y2, muted, label=None, dash=False):
    head = 7.0
    dx, dy = x2 - x1, y2 - y1
    ln = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    ux, uy = dx / ln, dy / ln
    ex, ey = x2 - ux * head, y2 - uy * head
    px, py = -uy, ux
    dash_attr = ' stroke-dasharray="4 3"' if dash else ""
    s = (
        f'  <line x1="{x1}" y1="{y1}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{muted}" '
        f'stroke-width="1.6"{dash_attr}/>\n'
        f'  <polygon points="{x2},{y2} {ex + px * 4.4:.1f},{ey + py * 4.4:.1f} '
        f'{ex - px * 4.4:.1f},{ey - py * 4.4:.1f}" fill="{muted}"/>\n'
    )
    if label:
        s += (
            f'  <text class="cap" x="{(x1 + x2) / 2}" y="{(y1 + y2) / 2 - 7}" '
            f'text-anchor="middle">{label}</text>\n'
        )
    return s


def write(svg, name):
    for chunk in svg.split("<text")[1:]:
        body = chunk.split(">", 1)[1].split("</text>")[0]
        assert "\n" not in body, f"{name}: newline inside a text element"
    (OUT / name).write_text(svg, encoding="utf-8", newline="\n")
    print(f"  wrote {name}")


def render(theme: str, name: str) -> None:
    p = PALETTES[theme]
    W, H = 1080, 540
    s = header(
        W,
        H,
        "Train, promote, monitor, serve",
        "A challenger ships only if it measurably beats both the training mean and "
        "the current champion. Drift decides when to try.",
        p,
    )

    s += '  <text class="lane" x="30" y="96">Training &amp; promotion — scheduled</text>\n'
    s += box(30, 108, 170, 74, p["grey_bg"], p["hair"], "UCI hourly data", ["SHA-256 pinned"])
    s += arrow(204, 145, 236, 145, p["muted"])
    s += box(240, 108, 170, 74, p["blue_bg"], p["blue"], "Train", ["two gates"])
    s += arrow(414, 145, 446, 145, p["muted"])
    s += box(
        450, 108, 170, 74, p["green_bg"], p["green"], "MLflow registry", ["every challenger logged"]
    )
    s += arrow(624, 145, 676, 145, p["muted"], "if better")
    s += box(680, 108, 160, 74, p["amber_bg"], p["amber"], "@champion", ["the live pointer"])

    s += '  <text class="lane" x="30" y="232">Monitoring — after each window</text>\n'
    s += box(
        240, 244, 220, 70, p["amber_bg"], p["amber"], "Drift check", ["target vs. feature drift"]
    )
    s += arrow(327, 244, 327, 185, p["muted"], "retrain")

    s += '  <text class="lane" x="30" y="356">Serving — online, per request</text>\n'
    s += box(30, 368, 140, 74, p["grey_bg"], p["hair"], "Request", [])
    s += arrow(174, 405, 206, 405, p["muted"])
    s += box(210, 368, 180, 74, p["blue_bg"], p["blue"], "FastAPI service", ["live before ready"])
    s += arrow(394, 405, 426, 405, p["muted"])
    s += box(430, 368, 150, 74, p["grey_bg"], p["hair"], "Prediction", ["~2ms warm"])
    s += arrow(584, 405, 616, 405, p["muted"])
    s += box(620, 368, 150, 74, p["grey_bg"], p["hair"], "Prometheus", [])
    s += arrow(774, 405, 806, 405, p["muted"])
    s += box(810, 368, 150, 74, p["grey_bg"], p["hair"], "Grafana", [])

    # Cross-connections: the champion feeds serving, and serving feeds monitoring.
    # Routed to x=370 rather than straight into FastAPI's centre so the line
    # passes to the right of the Drift check box instead of clipping its corner.
    s += arrow(730, 182, 370, 368, p["faint"], None, dash=True)
    s += '  <text class="cap" x="560" y="268" text-anchor="middle">loads @champion</text>\n'
    s += arrow(505, 368, 360, 314, p["faint"], None, dash=True)
    s += '  <text class="cap" x="500" y="344" text-anchor="middle">serving traffic log</text>\n'

    s += (
        f'  <line x1="30" y1="462" x2="{W - 30}" y2="462" stroke="{p["hair"]}" stroke-width="1"/>\n'
    )
    s += '  <text class="lbl" x="30" y="488">Two gates decide whether a model ships</text>\n'
    for i, ln in enumerate(
        [
            "A challenger must beat the training-mean baseline by 35%, which catches a "
            "broken dataset or a model that learned nothing.",
            "Then it must beat the current champion, re-scored on the challenger's own "
            "validation window, by 2% — comparing stored metrics across windows of "
            "different difficulty is meaningless.",
        ]
    ):
        s += f'  <text class="cap" x="30" y="{510 + i * 19}">{ln}</text>\n'
    s += "</svg>\n"
    write(s, name)


def main() -> int:
    render("light", "architecture.svg")
    render("dark", "architecture-dark.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
