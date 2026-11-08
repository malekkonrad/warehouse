"""Generate PNG sprites for the warehouse visualisation.

Run once:  uv run python warehouse/render/make_assets.py
"""
import os
from PIL import Image, ImageDraw

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


def make_robot(size: int = 256) -> None:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img, "RGBA")

    cx, cy = size / 2, size / 2

    # Wheels (left + right, drawn behind chassis)
    wheel_w, wheel_h = size * 0.16, size * 0.42
    for sx in (-1, 1):
        x0 = cx + sx * size * 0.34 - wheel_w / 2
        y0 = cy - wheel_h / 2
        d.rounded_rectangle(
            [x0, y0, x0 + wheel_w, y0 + wheel_h],
            radius=size * 0.04,
            fill=(30, 30, 35, 255),
        )
        for ty in (0.25, 0.5, 0.75):
            d.line(
                [(x0 + wheel_w * 0.2, y0 + wheel_h * ty),
                 (x0 + wheel_w * 0.8, y0 + wheel_h * ty)],
                fill=(70, 70, 75, 255), width=2,
            )

    # Chassis
    body_w, body_h = size * 0.62, size * 0.78
    bx0, by0 = cx - body_w / 2, cy - body_h / 2
    bx1, by1 = bx0 + body_w, by0 + body_h
    d.rounded_rectangle(
        [bx0, by0, bx1, by1],
        radius=size * 0.09,
        fill=(70, 110, 180, 255),
        outline=(25, 45, 85, 255),
        width=3,
    )

    # Top highlight strip
    d.rounded_rectangle(
        [bx0 + body_w * 0.10, by0 + body_h * 0.55,
         bx1 - body_w * 0.10, by0 + body_h * 0.78],
        radius=size * 0.04,
        fill=(95, 140, 210, 255),
    )

    # Front (image-top) hazard stripes
    front_h = size * 0.10
    n = 6
    sw = body_w / n
    for i in range(n):
        col = (255, 200, 0, 255) if i % 2 == 0 else (20, 20, 20, 255)
        d.rectangle(
            [bx0 + i * sw, by0, bx0 + (i + 1) * sw, by0 + front_h],
            fill=col,
        )
    d.rectangle(
        [bx0, by0, bx1, by0 + front_h],
        outline=(20, 20, 20, 255), width=2,
    )

    # Sensor "head" (slightly behind front)
    head_r = size * 0.13
    hcx, hcy = cx, cy + size * 0.04
    d.ellipse(
        [hcx - head_r, hcy - head_r, hcx + head_r, hcy + head_r],
        fill=(35, 35, 40, 255),
        outline=(15, 15, 20, 255), width=2,
    )
    lens_r = head_r * 0.55
    d.ellipse(
        [hcx - lens_r, hcy - lens_r, hcx + lens_r, hcy + lens_r],
        fill=(80, 180, 220, 255),
        outline=(25, 75, 105, 255), width=2,
    )
    d.ellipse(
        [hcx - lens_r * 0.45, hcy - lens_r * 0.55,
         hcx + lens_r * 0.05, hcy - lens_r * 0.05],
        fill=(225, 240, 255, 255),
    )

    # Antenna with red light
    ant_top = by0 - size * 0.05
    d.line([(cx, by0 + size * 0.005), (cx, ant_top)],
           fill=(40, 40, 40, 255), width=2)
    r = size * 0.025
    d.ellipse([cx - r, ant_top - r, cx + r, ant_top + r],
              fill=(220, 60, 60, 255))

    img.save(os.path.join(ASSETS_DIR, "robot.png"))


def make_package(size: int = 128) -> None:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img, "RGBA")

    m = size * 0.06
    box = [m, m, size - m, size - m]

    # Cardboard fill
    d.rectangle(box, fill=(193, 154, 107, 255),
                outline=(110, 75, 40, 255), width=3)

    # Subtle grain
    for y in range(int(m) + 4, int(size - m), 6):
        d.line([(m + 3, y), (size - m - 3, y)],
               fill=(215, 180, 135, 70), width=1)

    # Center seam
    seam_y = size / 2
    d.line([(m, seam_y), (size - m, seam_y)],
           fill=(95, 65, 30, 255), width=2)

    # Tape across seam
    tape_h = size * 0.16
    d.rectangle(
        [m, seam_y - tape_h / 2, size - m, seam_y + tape_h / 2],
        fill=(238, 218, 158, 230),
        outline=(180, 158, 100, 220), width=1,
    )

    # Diagonal corner shadow for depth
    shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.polygon(
        [(size - m, m), (size - m, size - m), (m, size - m)],
        fill=(0, 0, 0, 35),
    )
    img = Image.alpha_composite(img, shadow)

    img.save(os.path.join(ASSETS_DIR, "package.png"))


def make_shelf(size: int = 128) -> None:
    """Tileable 1 m × 1 m shelf segment, viewed top-down."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img, "RGBA")

    # Frame
    d.rectangle([0, 0, size - 1, size - 1], fill=(85, 85, 92, 255))

    # Inner cargo bay
    inset = size * 0.13
    d.rectangle(
        [inset, inset, size - inset, size - inset],
        fill=(55, 55, 62, 255),
    )

    # 2×2 grid of stored crates
    crate_colors = [
        (180, 90, 60, 255), (155, 120, 70, 255),
        (95, 135, 95, 255), (165, 150, 70, 255),
    ]
    pad = size * 0.18
    gap = size * 0.04
    bw = (size - 2 * pad - gap) / 2
    for i, (row, col) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        x0 = pad + col * (bw + gap)
        y0 = pad + row * (bw + gap)
        d.rectangle(
            [x0, y0, x0 + bw, y0 + bw],
            fill=crate_colors[i],
            outline=(25, 25, 25, 255), width=1,
        )
        # tape detail on each crate
        d.line(
            [(x0, y0 + bw / 2), (x0 + bw, y0 + bw / 2)],
            fill=(230, 215, 170, 180), width=1,
        )

    # Corner posts (vertical supports)
    post = size * 0.11
    for (x, y) in [(0, 0), (size - post, 0),
                   (0, size - post), (size - post, size - post)]:
        d.rectangle([x, y, x + post, y + post], fill=(28, 28, 32, 255))

    # Yellow safety border
    edge = max(2, int(size * 0.02))
    d.rectangle([0, 0, size - 1, edge], fill=(230, 190, 30, 255))
    d.rectangle([0, size - 1 - edge, size - 1, size - 1],
                fill=(230, 190, 30, 255))
    d.rectangle([0, 0, edge, size - 1], fill=(230, 190, 30, 255))
    d.rectangle([size - 1 - edge, 0, size - 1, size - 1],
                fill=(230, 190, 30, 255))

    img.save(os.path.join(ASSETS_DIR, "shelf.png"))


if __name__ == "__main__":
    make_robot()
    make_package()
    make_shelf()
    print("Assets written to:", ASSETS_DIR)
