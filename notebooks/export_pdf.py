"""Eksport notebooka raportu do PDF z **montażami klatek** zamiast GIF-ów.

PDF jest formatem statycznym — animacja GIF-a w nim nie zagra, a silniki LaTeX
nie czytają formatu GIF w ogóle. Dlatego z każdego GIF-a wycinamy kilka
równomiernie rozłożonych klatek i sklejamy w jeden statyczny obraz (montaż),
który w druku pokazuje trajektorię agentów. Następnie:

  notebook → kopia z podmienionymi outputami (gif → png montaż) → pandoc → PDF

Uruchomienie:
    uv run python warehouse/report/export_pdf.py

Wynik: warehouse/report/projekt6_multiagent.pdf
"""
from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
NB_IN = HERE / "projekt6_multiagent.ipynb"
NB_EXPORT = HERE / "_export_projekt6_multiagent.ipynb"
PDF_OUT = HERE / "projekt6_multiagent.pdf"
GIFS = HERE / "gifs"

# Kolejność GIF-ów w outputach notebooka (homo loop wg RUNS_HOMO, potem hetero).
GIF_ORDER = ["homo_ppo_v1", "homo_ppo_v2", "homo_sac_v1", "hetero"]
TITLES = {
    "homo_ppo_v1": "Homo PPO-v1",
    "homo_ppo_v2": "Homo PPO-v2",
    "homo_sac_v1": "Homo SAC",
    "hetero": "Hetero (PPO-v1 + PPO-v2 + SAC)",
}

N_FRAMES = 6        # ile klatek w montażu
COLS = 3            # układ siatki
THUMB = 280         # szerokość pojedynczej miniatury (px)
PAD = 6             # odstęp między klatkami (px)


BAR = 34  # wysokość paska z tytułem (px)


def _font(size: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def make_montage(gif_path: Path, title: str) -> bytes:
    """Zwraca PNG (bytes) — pasek z tytułem + siatka N_FRAMES klatek."""
    im = Image.open(gif_path)
    n = getattr(im, "n_frames", 1)
    idxs = [round(k * (n - 1) / (N_FRAMES - 1)) for k in range(N_FRAMES)]
    thumbs = []
    for i in idxs:
        im.seek(i)
        f = im.convert("RGB")
        ratio = THUMB / f.width
        f = f.resize((THUMB, round(f.height * ratio)))
        thumbs.append(f)
    rows = (len(thumbs) + COLS - 1) // COLS
    cw, ch = thumbs[0].width, thumbs[0].height
    W = COLS * cw + (COLS + 1) * PAD
    H = BAR + rows * ch + (rows + 1) * PAD
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((PAD, 7), title, fill="black", font=_font(22))
    for k, t in enumerate(thumbs):
        r, c = divmod(k, COLS)
        x = PAD + c * (cw + PAD)
        y = BAR + PAD + r * (ch + PAD)
        canvas.paste(t, (x, y))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def main():
    if not NB_IN.exists():
        sys.exit(f"Brak notebooka: {NB_IN}")

    # 1. Montaże dla każdego GIF-a
    montages: dict[str, str] = {}  # name -> base64 png
    for name in GIF_ORDER:
        gp = GIFS / f"{name}.gif"
        if not gp.exists():
            sys.exit(f"Brak GIF-a: {gp}")
        png = make_montage(gp, TITLES[name])
        montages[name] = base64.b64encode(png).decode("ascii")
        (GIFS / f"montage_{name}.png").write_bytes(base64.b64decode(montages[name]))
        print(f"montaż: {name}  ({len(png)//1024} KB)")

    # 2. Kopia notebooka z podmienionymi outputami gif -> png montaż
    nb = json.loads(NB_IN.read_text())
    gif_outputs = []
    for c in nb["cells"]:
        if c.get("cell_type") != "code":
            continue
        for o in c.get("outputs", []):
            if "image/gif" in o.get("data", {}):
                gif_outputs.append(o)
    if len(gif_outputs) != len(GIF_ORDER):
        sys.exit(
            f"Spodziewano się {len(GIF_ORDER)} outputów gif, znaleziono "
            f"{len(gif_outputs)} — zaktualizuj GIF_ORDER."
        )
    for o, name in zip(gif_outputs, GIF_ORDER):
        data = o["data"]
        del data["image/gif"]
        data["image/png"] = montages[name]
        data["text/plain"] = [f"[montaż klatek: {name}]"]

    # display(Markdown(...)) zostawia też text/plain "<IPython...object>",
    # które pandoc wyświetla zamiast właściwego markdownu -> usuwamy je.
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            data = o.get("data", {})
            if "text/markdown" in data and "text/plain" in data:
                del data["text/plain"]
    # pdflatex obsługuje polskie litery (inputenc), ale wykłada się na
    # symbolach matematycznych/strzałkach/emoji -> sanityzacja do ASCII.
    SANITIZE = {
        "→": " -> ", "↔": " <-> ", "⇒": " => ",
        "≈": " ~= ", "≫": " >> ", "≪": " << ",
        "≥": " >= ", "≤": " <= ", "≠": " != ",
        "×": "x", "·": "*", "−": "-", "–": "-", "—": " - ",
        "„": '"', "”": '"', "“": '"', "’": "'", "‘": "'", "…": "...",
        "✅": "[OK]", "❌": "[nie]", "⚠️": "[!]", "⚠": "[!]",
        "Δ": "d", "≥": " >= ",
        # znaki ramki z dekoracyjnych komentarzy w komórkach kodu
        "─": "-", "│": "|", "┌": "+", "┐": "+", "└": "+", "┘": "+",
    }
    for c in nb["cells"]:
        if c.get("cell_type") not in ("markdown", "code"):
            continue
        src = c["source"]
        joined = "".join(src) if isinstance(src, list) else src
        for k, v in SANITIZE.items():
            joined = joined.replace(k, v)
        c["source"] = joined
        # ostrzeż o pozostałych znakach spoza Latin Extended-A (Polski mieści się)
        leftover = sorted({ch for ch in joined if ord(ch) > 0x17F})
        if leftover:
            print(f"  UWAGA nieobsłużone znaki ({c['cell_type']}): {leftover}")

    NB_EXPORT.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print(f"kopia eksportowa: {NB_EXPORT.name}")

    # 3. pandoc -> PDF (pdflatex; polskie litery przez inputenc/babel)
    cmd = [
        "pandoc", str(NB_EXPORT), "-o", str(PDF_OUT),
        "--pdf-engine=pdflatex",
        "-V", "lang=pl",
        "-V", "geometry:margin=2cm",
        "--wrap=preserve",
    ]
    print("pandoc:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:], file=sys.stderr)
        sys.exit("pandoc nie zbudował PDF — patrz log wyżej.")
    print(f"\nOK → {PDF_OUT}  ({PDF_OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
