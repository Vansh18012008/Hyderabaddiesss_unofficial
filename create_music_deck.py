from pathlib import Path
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parent
PIPE = ROOT / "joint_music_pipeline"
OUT = ROOT / "joint_music_music_pipeline_results.pptx"

W, H = 13.333, 7.5
BG = RGBColor(10, 22, 38)
PANEL = RGBColor(20, 38, 59)
PANEL2 = RGBColor(28, 51, 75)
WHITE = RGBColor(242, 247, 252)
MUTED = RGBColor(172, 190, 207)
TEAL = RGBColor(53, 211, 190)
ORANGE = RGBColor(255, 178, 76)
PINK = RGBColor(255, 91, 178)
RED = RGBColor(255, 110, 110)
GREEN = RGBColor(104, 225, 160)


def rgb(hex_value):
    hex_value = hex_value.lstrip("#")
    return RGBColor.from_string(hex_value)


def box(slide, x, y, w, h, fill=PANEL, line=None, radius=False):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    s = slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.color.rgb = line or fill
    if radius:
        s.adjustments[0] = 0.08
    return s


def text(slide, value, x, y, w, h, size=18, color=WHITE, bold=False,
         align=PP_ALIGN.LEFT, font="Aptos", margin=0.06, valign=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(margin)
    tf.margin_right = Inches(margin)
    tf.margin_top = Inches(margin)
    tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = value
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return tb


def rich_text(slide, parts, x, y, w, h, size=18, align=PP_ALIGN.LEFT, margin=0.06):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(margin)
    tf.margin_right = Inches(margin)
    tf.margin_top = Inches(margin)
    tf.margin_bottom = Inches(margin)
    p = tf.paragraphs[0]
    p.alignment = align
    for value, color, bold in parts:
        r = p.add_run()
        r.text = value
        r.font.name = "Aptos"
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
    return tb


def bullet_list(slide, items, x, y, w, h, size=18, color=WHITE, bullet_color=TEAL, gap=0.03):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(0.04)
    tf.margin_right = Inches(0.03)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = item
        p.level = 0
        p.font.name = "Aptos"
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.space_after = Pt(7)
        p.bullet = True
    return tb


def title(slide, number, heading, subtitle=None):
    text(slide, f"{number:02d}", 0.45, 0.28, 0.55, 0.38, size=16, color=TEAL, bold=True)
    text(slide, heading, 1.08, 0.23, 11.6, 0.48, size=27, color=WHITE, bold=True)
    if subtitle:
        text(slide, subtitle, 1.1, 0.78, 11.5, 0.35, size=12, color=MUTED)
    box(slide, 0.45, 1.12, 12.43, 0.015, fill=rgb("#2e4a68"))


def source(slide, value):
    text(slide, f"Source: {value}", 0.52, 7.15, 12.1, 0.18, size=8.5, color=MUTED)


def add_image(slide, path, x, y, w, h):
    from PIL import Image
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(w / iw, h / ih)
    nw, nh = iw * scale, ih * scale
    slide.shapes.add_picture(str(path), Inches(x + (w - nw) / 2), Inches(y + (h - nh) / 2),
                             width=Inches(nw), height=Inches(nh))


def arrow(slide, x1, y1, x2, y2, color=TEAL, width=2):
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                                      Inches(x2), Inches(y2))
    line.line.color.rgb = color
    line.line.width = Pt(width)
    line.line.end_arrowhead = True
    return line


def new_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = BG
    return slide


def stat_card(slide, value, label, x, y, w=2.2, color=TEAL):
    box(slide, x, y, w, 1.02, fill=PANEL, line=rgb("#345474"), radius=True)
    text(slide, value, x + 0.1, y + 0.12, w - 0.2, 0.42, size=25, color=color, bold=True, align=PP_ALIGN.CENTER)
    text(slide, label, x + 0.1, y + 0.6, w - 0.2, 0.28, size=11, color=MUTED, align=PP_ALIGN.CENTER)


def table(slide, headers, rows, x, y, widths, row_h=0.5, font=12, header_fill=rgb("#24516a")):
    total_w = sum(widths)
    for j, h in enumerate(headers):
        box(slide, x + sum(widths[:j]), y, widths[j], row_h, fill=header_fill, line=BG)
        text(slide, h, x + sum(widths[:j]) + 0.06, y + 0.08, widths[j] - 0.12, row_h - 0.1,
             size=font, color=WHITE, bold=True, valign=MSO_ANCHOR.MIDDLE)
    for i, row in enumerate(rows):
        yy = y + row_h * (i + 1)
        for j, cell in enumerate(row):
            fill = PANEL if i % 2 == 0 else PANEL2
            box(slide, x + sum(widths[:j]), yy, widths[j], row_h, fill=fill, line=BG)
            text(slide, str(cell), x + sum(widths[:j]) + 0.06, yy + 0.06, widths[j] - 0.12, row_h - 0.08,
                 size=font, color=WHITE if j != 0 else ORANGE, bold=(j == 0),
                 valign=MSO_ANCHOR.MIDDLE)
    return total_w


def build():
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)

    # 1
    s = new_slide(prs)
    box(s, 0, 0, W, H, fill=BG)
    text(s, "JOINT 2D MUSIC", 0.7, 0.75, 7.4, 0.55, size=35, color=TEAL, bold=True)
    text(s, "Range–angle localization on\nRATHAN_WALKING_VIDEO.bin", 0.7, 1.45, 8.8, 1.35, size=30, color=WHITE, bold=True)
    text(s, "A reproducible, threshold-explicit walkthrough of the method, outputs, and limits", 0.74, 3.05, 8.2, 0.4, size=17, color=MUTED)
    box(s, 0.72, 4.05, 5.7, 1.18, fill=PANEL, line=rgb("#345474"), radius=True)
    text(s, "Core claim", 0.95, 4.25, 1.3, 0.25, size=13, color=ORANGE, bold=True)
    text(s, "A MUSIC peak is a possible reflection location — not proof that a person is there.", 0.95, 4.55, 5.1, 0.48, size=18, color=WHITE, bold=True)
    text(s, "Joint MUSIC models range and angle together; downstream grouping and presence gates decide what is displayed.", 0.75, 6.35, 10.2, 0.35, size=13, color=MUTED)
    text(s, "Technical viva deck  |  current project output", 0.75, 7.12, 8.0, 0.18, size=9, color=MUTED)
    source(s, "joint_music_pipeline/README.md; output/REPORT.md")

    # 2
    s = new_slide(prs)
    title(s, 2, "What the BIN contains", "The decoder treats the recording as complex baseband I/Q samples")
    # cube
    box(s, 0.75, 1.55, 4.9, 4.75, fill=PANEL, line=rgb("#345474"), radius=True)
    text(s, "Decoded data cube", 1.0, 1.82, 4.3, 0.3, size=20, color=TEAL, bold=True)
    # axes
    for i in range(4):
        box(s, 1.55 + i * 0.65, 2.75, 0.45, 1.9, fill=rgb("#1d8d85"), line=TEAL)
        text(s, f"RX {i+1}", 1.42 + i * 0.65, 4.83, 0.7, 0.22, size=10, color=WHITE, align=PP_ALIGN.CENTER)
    text(s, "4 receivers", 1.0, 5.2, 3.9, 0.3, size=16, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    arrow(s, 3.95, 2.45, 3.95, 2.68, ORANGE, 2)
    text(s, "1,024 chirps / slow time", 2.4, 2.0, 2.8, 0.28, size=14, color=ORANGE, bold=True, align=PP_ALIGN.CENTER)
    text(s, "512 complex ADC samples / chirp\nfast time →", 1.0, 5.6, 4.2, 0.48, size=14, color=MUTED, align=PP_ALIGN.CENTER)
    box(s, 6.05, 1.55, 6.45, 1.22, fill=PANEL2, line=rgb("#345474"), radius=True)
    text(s, "Complex I/Q", 6.35, 1.78, 1.8, 0.3, size=20, color=ORANGE, bold=True)
    text(s, "Each sample carries an in-phase (I) and quadrature (Q) component: a 2D snapshot of the received sinusoid.", 8.0, 1.72, 4.1, 0.55, size=15, color=WHITE)
    box(s, 6.05, 3.05, 6.45, 1.28, fill=PANEL2, line=rgb("#345474"), radius=True)
    text(s, "Fast time", 6.35, 3.3, 1.65, 0.3, size=19, color=TEAL, bold=True)
    text(s, "Samples within one chirp. Their phase progression is tied to beat frequency → provisional range.", 8.0, 3.22, 4.1, 0.63, size=15, color=WHITE)
    box(s, 6.05, 4.65, 6.45, 1.28, fill=PANEL2, line=rgb("#345474"), radius=True)
    text(s, "Slow time", 6.35, 4.9, 1.65, 0.3, size=19, color=TEAL, bold=True)
    text(s, "Successive chirps. Phase changes across chirps carry motion and, later, possible vital-rate periodicity.", 8.0, 4.82, 4.1, 0.63, size=15, color=WHITE)
    source(s, "adc_io.py; output/radar_parameter_derivations.json; output/summary.json")

    # 3
    s = new_slide(prs)
    title(s, 3, "Parameter mapping from the supplied notebook", "Radar constants are inputs; the processing method remains our joint MUSIC + phase estimator")
    rows = [
        ("ADC", "512 samples/chirp · 4 RX · 16-bit complex I/Q · 600 ksample/s"),
        ("Sweep", "77–79.982 GHz · 100 μs ramp · 2.982 GHz bandwidth"),
        ("Derived", "29.82 THz/s slope · 78.491 GHz center frequency"),
        ("RATHAN", "1,024 chirps decoded; do not truncate to the notebook example’s 512"),
        ("Timebase", "20 chirps/s inferred from notebook axis; not independently verified for RATHAN"),
    ]
    table(s, ["Item", "Applied / caveated value"], rows, 0.7, 1.5, [1.55, 11.65], row_h=0.56, font=14)
    box(s, 0.7, 4.65, 5.85, 1.55, fill=PANEL, line=ORANGE, radius=True)
    text(s, "Calibration warnings", 0.95, 4.88, 2.5, 0.25, size=17, color=ORANGE, bold=True)
    bullet_list(s, [
        "512 / 600 ksample/s = 0.853 ms, conflicting with the stated 0.100 ms ramp.",
        "Notebook literal λmax/2 × 250 = 0.487 m is rejected as a units/expression error.",
        "Run assumes half-wavelength RX spacing = 1.948 mm."
    ], 0.96, 5.17, 5.25, 0.85, size=12.5)
    box(s, 6.8, 4.65, 5.55, 1.55, fill=PANEL, line=TEAL, radius=True)
    text(s, "Provisional range numbers", 7.05, 4.88, 3.0, 0.25, size=17, color=TEAL, bold=True)
    text(s, "ADC-limited max ≈ 1.509 m\nFFT bin spacing ≈ 5.895 mm\nBandwidth resolution ≈ 5.03 cm", 7.05, 5.2, 4.85, 0.78, size=15, color=WHITE)
    text(s, "Bin spacing is not the same as resolving two physical targets.", 7.05, 6.05, 4.85, 0.25, size=11.5, color=MUTED)
    source(s, "config.json; output/radar_parameter_derivations.json; README.md")

    # 4
    s = new_slide(prs)
    title(s, 4, "End-to-end method sequence", "Each stage has a distinct role; only the joint MUSIC map is the main range–angle spectrum")
    stages = [
        ("1", "BIN decode", "complex cube"),
        ("2", "Preprocess", "per-chirp DC + slow-time mean"),
        ("3", "Window", "32 chirps; hop 16"),
        ("4", "Smooth", "spatial + fast-time apertures"),
        ("5", "Covariance", "FB averaging"),
        ("6", "Joint MUSIC", "range × angle map"),
        ("7", "Detect", "local peaks + energy"),
        ("8", "Group / track", "candidate support"),
        ("9", "Presence gate", "display or blank"),
    ]
    x0 = 0.55
    for i, (n, head, sub) in enumerate(stages):
        x = x0 + (i % 5) * 2.52
        y = 1.65 + (i // 5) * 2.05
        box(s, x, y, 2.1, 1.15, fill=PANEL2 if i < 6 else PANEL, line=TEAL if i == 5 else rgb("#345474"), radius=True)
        text(s, n, x + 0.14, y + 0.14, 0.32, 0.32, size=16, color=ORANGE, bold=True)
        text(s, head, x + 0.52, y + 0.13, 1.43, 0.3, size=14, color=WHITE, bold=True)
        text(s, sub, x + 0.16, y + 0.57, 1.8, 0.38, size=11, color=MUTED)
        if i not in (4, 8):
            arrow(s, x + 2.12, y + 0.57, x + 2.46, y + 0.57, TEAL, 1.5)
    box(s, 0.85, 6.0, 11.65, 0.6, fill=rgb("#2e2534"), line=PINK, radius=True)
    text(s, "Important: FFT range energy is auxiliary for detection; a range FFT is also used later for vital-signal extraction. The displayed map is generated by joint MUSIC, not FFT → CBF.", 1.1, 6.18, 11.1, 0.25, size=13, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    source(s, "README.md; run_pipeline.py; joint_music.py")

    # 5
    s = new_slide(prs)
    title(s, 5, "Intuitive explanation of 2D MUSIC", "Try every possible location’s expected echo pattern, then measure how well it fits")
    box(s, 0.7, 1.55, 5.5, 4.8, fill=PANEL, line=rgb("#345474"), radius=True)
    text(s, "The analogy", 1.0, 1.85, 2.0, 0.3, size=21, color=ORANGE, bold=True)
    text(s, "Imagine a wall of tuning forks. For each trial range–angle location, we predict the phase pattern that should appear across receivers and fast-time samples.", 1.0, 2.35, 4.8, 1.15, size=19, color=WHITE)
    text(s, "If the measured samples match that pattern, the trial location “lights up.” If they do not, it stays dark.", 1.0, 3.8, 4.8, 0.85, size=19, color=TEAL, bold=True)
    text(s, "This is localization of a signal-like return — not recognition of a person.", 1.0, 5.35, 4.8, 0.5, size=16, color=MUTED)
    # diagram
    box(s, 6.65, 1.65, 5.85, 4.5, fill=PANEL2, line=rgb("#345474"), radius=True)
    text(s, "Subspace view", 6.95, 1.95, 4.8, 0.3, size=20, color=TEAL, bold=True)
    # receiver dots
    for i in range(4):
        box(s, 7.25 + i * 0.62, 3.1, 0.22, 0.22, fill=ORANGE, line=ORANGE)
        text(s, f"RX{i+1}", 7.0 + i * 0.62, 3.42, 0.7, 0.2, size=9, color=MUTED, align=PP_ALIGN.CENTER)
    for i in range(4):
        arrow(s, 7.35 + i * 0.62, 3.0, 9.0, 4.35, rgb("#5f7c9a"), 1)
    box(s, 8.75, 4.15, 2.0, 0.75, fill=rgb("#1d8d85"), line=TEAL, radius=True)
    text(s, "trial (r, θ)", 9.0, 4.38, 1.5, 0.25, size=18, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    text(s, "Covariance eigenvectors split the data into a signal subspace and a noise subspace. MUSIC scores trial steering vectors that are nearly orthogonal to the noise subspace.", 6.95, 5.1, 5.05, 0.78, size=14, color=WHITE)
    text(s, "Range: beat-frequency / fast-time phase structure\nAngle: phase differences across RX elements", 6.95, 2.48, 5.0, 0.58, size=14, color=ORANGE, bold=True)
    source(s, "joint_music.py: steering_matrix(), spectrum_from_steering()")

    # 6
    s = new_slide(prs)
    title(s, 6, "MUSIC settings and the rendered map", "Settings are explicit and reproducible; “rank 6” is a subspace choice, not six people")
    rows = [
        ("Windows", "32 chirps; hop 16 → 63 overlapping windows"),
        ("At assumed 20 Hz", "≈1.6 s window; ≈0.8 s advance"),
        ("Smoothing", "3 RX aperture; 64 fast-time samples; fast-time step 8"),
        ("Covariance", "192 dimensions; forward–backward averaging enabled"),
        ("Scan", "−60°…+60° in 1° steps; equivalent bins 0…255"),
        ("Rank", "6 signal-subspace components; not a person count"),
    ]
    table(s, ["Setting", "Current run"], rows, 0.65, 1.43, [1.55, 5.2], row_h=0.48, font=11.7)
    img = PIPE / "output" / "frames" / "frame_000.png"
    box(s, 6.85, 1.43, 5.75, 4.8, fill=PANEL, line=rgb("#345474"), radius=True)
    add_image(s, img, 7.0, 1.62, 5.45, 4.2)
    text(s, "Actual rendered frame 1 / 63", 7.05, 5.86, 5.1, 0.22, size=11, color=MUTED, align=PP_ALIGN.CENTER)
    box(s, 0.65, 6.2, 5.95, 0.52, fill=rgb("#2e2534"), line=ORANGE, radius=True)
    text(s, "Color scale is per-window normalized: brightest map cell = 0 dB.", 0.85, 6.36, 5.55, 0.2, size=12.5, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    source(s, "config.json; joint_music.py; output/frames/frame_000.png")

    # 7
    s = new_slide(prs)
    title(s, 7, "Detection thresholds: separate stages, separate meanings", "Thresholds are run configuration—not universal radar laws")
    rows = [
        ("Global contrast", "≥ 6 dB peak-to-median", "Run local-peak detection; otherwise no raw peaks"),
        ("Local maximum", "3 angle × 7 range-bin neighborhood", "Keep only neighborhood maxima"),
        ("Peak level", "≥ −22 dB vs frame max", "Reject weak map cells"),
        ("Range excess", "≥ 4 dB over median range energy", "Reject peaks without auxiliary range support"),
        ("Exclusions", "Bins < 8, final bin, angle edges", "Reject scan-edge / near-DC candidates"),
        ("NMS", "≤5 bins and ≤10°", "Treat as duplicate; keep the stronger peak"),
        ("Cap", "≤24 peaks/window", "Stop after the top 24 accepted peaks"),
        ("Strong", "map ≥−15 dB; excess ≥8 dB; contrast ≥8 dB", "Eligible to start / confirm tracks"),
    ]
    table(s, ["Stage", "Threshold", "When it fails"], rows, 0.5, 1.42, [1.75, 4.05, 6.95], row_h=0.5, font=11.3)
    box(s, 0.75, 6.0, 11.85, 0.57, fill=rgb("#2e2534"), line=PINK, radius=True)
    text(s, "Support labels—not probabilities and not person identities.", 1.0, 6.18, 11.35, 0.2, size=13, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    source(s, "walking_confidence_tracker.py; group_output.py; output/settings.json")

    # 8
    s = new_slide(prs)
    title(s, 8, "Grouping and tracking thresholds", "The pipeline tries to avoid turning every bright return into a person")
    rows = [
        ("Group proximity", "≤18 range bins and ≤12°", "Peaks may be compatible"),
        ("Valley separation", "≥8 dB immediately; ≥4 dB for 2 successive windows", "Keep peaks separate"),
        ("Motion separation", "After 3 observations: >4 bins/window or >4°/window", "Keep different motions separate"),
        ("Linkage", "Complete-link grouping", "No arbitrarily long chain of near neighbors"),
        ("Confirmation", "3 consecutive strong matches", "Track becomes confirmed"),
        ("Weak continuation", "Weak cannot start/confirm; ≤5 windows", "Can only continue confirmed track"),
        ("Retirement", ">2 missing windows", "Track is retired"),
    ]
    table(s, ["Rule", "Exact setting", "Effect"], rows, 0.55, 1.42, [2.05, 4.15, 6.65], row_h=0.56, font=11.8)
    box(s, 0.75, 5.82, 5.7, 0.72, fill=PANEL, line=TEAL, radius=True)
    text(s, "Single-person prior is configured (expected_people = 1).", 1.0, 6.02, 5.2, 0.22, size=14, color=TEAL, bold=True, align=PP_ALIGN.CENTER)
    box(s, 6.75, 5.82, 5.75, 0.72, fill=PANEL, line=ORANGE, radius=True)
    text(s, "Group IDs are reflector tracks, not verified identities.", 7.0, 6.02, 5.25, 0.22, size=14, color=ORANGE, bold=True, align=PP_ALIGN.CENTER)
    source(s, "peak_grouping.py; walking_confidence_tracker.py; output/grouping_settings.json")

    # 9
    s = new_slide(prs)
    title(s, 9, "Final candidate-display threshold", "A separate map-level presence gate decides whether a circle is shown")
    # left early, right late
    early = PIPE / "output" / "frames" / "frame_000.png"
    late = PIPE / "output" / "frames" / "frame_062.png"
    for x, img, label, color in [(0.65, early, "Early: candidate shown", GREEN), (6.85, late, "Late: no candidate", RED)]:
        box(s, x, 1.48, 5.8, 4.5, fill=PANEL, line=color, radius=True)
        add_image(s, img, x + 0.15, 1.67, 5.5, 3.75)
        text(s, label, x + 0.25, 5.55, 5.3, 0.25, size=15, color=color, bold=True, align=PP_ALIGN.CENTER)
    box(s, 1.15, 6.23, 11.0, 0.62, fill=rgb("#2e2534"), line=ORANGE, radius=True)
    text(s, "Presence gate: map peak − map median ≥ 15 dB  |  first 48 windows pass; final 15 fail", 1.35, 6.42, 10.6, 0.2, size=14, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    text(s, "Because each map is normalized independently, 0 dB marks the brightest cell—not absolute power or SNR. A strong wall can pass; a weak person can fail.", 0.9, 5.98, 11.6, 0.2, size=10.5, color=MUTED, align=PP_ALIGN.CENTER)
    source(s, "group_output.py; output/REPORT.md; output/frames/frame_000.png and frame_062.png")

    # 10
    s = new_slide(prs)
    title(s, 10, "Results on RATHAN_WALKING_VIDEO.bin", "Current output only; no localization accuracy claim is possible without verified ground truth")
    stats = [
        ("63", "overlapping windows", TEAL),
        ("186", "raw peaks", ORANGE),
        ("133", "groups", TEAL),
        ("41", "multi-peak groups", ORANGE),
        ("6", "ambiguous group instances", PINK),
        ("10", "group-track segments", TEAL),
        ("13/63", "confirmed grouped candidate", ORANGE),
        ("48/63", "presence-gated candidate", GREEN),
    ]
    for i, (v, lab, c) in enumerate(stats):
        stat_card(s, v, lab, 0.6 + (i % 4) * 3.12, 1.52 + (i // 4) * 1.42, w=2.65, color=c)
    box(s, 0.75, 4.65, 5.7, 1.35, fill=PANEL, line=rgb("#345474"), radius=True)
    text(s, "Interpretation", 1.0, 4.88, 1.7, 0.25, size=18, color=TEAL, bold=True)
    text(s, "The run demonstrates a reproducible signal-processing pipeline and its explicit decisions. It does not demonstrate person-localization accuracy.", 1.0, 5.25, 5.05, 0.55, size=15, color=WHITE)
    box(s, 6.75, 4.65, 5.7, 1.35, fill=PANEL, line=rgb("#345474"), radius=True)
    text(s, "Replay", 7.0, 4.88, 1.2, 0.25, size=18, color=ORANGE, bold=True)
    text(s, "Interactive replay: joint_music_pipeline/output/replay.html\nDebug replay: joint_music_pipeline/output/debug_replay.html", 7.0, 5.25, 5.05, 0.55, size=14, color=WHITE)
    source(s, "output/summary.json; output/REPORT.md; output/replay.html")

    # 11
    s = new_slide(prs)
    title(s, 11, "BR/HR: downstream exploratory stage", "A rate needs repeated cycles over time; one localization frame cannot provide it")
    box(s, 0.65, 1.45, 4.1, 4.85, fill=PANEL, line=TEAL, radius=True)
    text(s, "What happens after localization", 0.95, 1.75, 3.45, 0.3, size=20, color=TEAL, bold=True)
    steps = [
        "Use candidate range/angle to beamform complex IQ over chirps",
        "Unwrap phase to obtain motion proxy",
        "Search repeated components in 0.1–0.5 Hz (BR) and 0.8–2.5 Hz (HR)",
        "Require a rolling history, not a single frame",
    ]
    bullet_list(s, steps, 0.95, 2.9, 3.35, 1.95, size=14)
    text(s, "Intuition: counting a rate is like timing several beats of a metronome.", 0.95, 5.4, 3.3, 0.45, size=14, color=ORANGE, bold=True)
    rows = [
        ("Strict gate", "20 s / 400 obs; ≥80% trusted support; one group; range span ≤6 bins; angle span ≤6°; stable amplitude; prominence ≥4 dB", "0 frames pass"),
        ("Why", "24 lack 20 s history; 39 fail tracked-signal continuity", "No quality-gated BR/HR"),
        ("Exploratory", "10–20 s history; bypasses quality gates", "36 frames (12–47)"),
        ("BR", "median 10.5/min", "range 6.0–23.9"),
        ("HR", "median 65.3/min", "range 57.9–73.1"),
    ]
    table(s, ["Stage", "Rule / value", "Current output"], rows, 5.05, 1.45, [1.35, 4.3, 2.25], row_h=0.68, font=10.6)
    box(s, 5.1, 5.78, 7.25, 0.58, fill=rgb("#4b321f"), line=ORANGE, radius=True)
    text(s, "UNVALIDATED: motion and candidate-cell changes may create these frequencies.", 5.3, 5.96, 6.85, 0.2, size=11.5, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    source(s, "vital_estimation.py; output/vital_summary.json; output/REPORT.md")

    # 12
    s = new_slide(prs)
    title(s, 12, "Viva cheat sheet: what is defensible?", "Answer the question directly, then state the assumption or limitation")
    qa = [
        ("What is a peak?", "A local maximum in the joint MUSIC map that also passes the configured level and auxiliary range-energy checks."),
        ("When is it rejected?", "If contrast is too low, it is not a local maximum, is too weak, lacks range-energy excess, is near an excluded edge, or is an NMS duplicate."),
        ("When is a candidate not shown?", "When the selected map peak-to-median contrast is below 15 dB. In this clip: 15/63 windows."),
        ("Does a candidate mean human?", "No. It means a signal-like reflector survived the pipeline’s heuristics; wall, multipath, clutter, motion, and weak echoes remain possible."),
        ("Why are strict BR/HR absent?", "No frame passed the 20 s quality-gated tracked-signal continuity checks: 24 lacked history and 39 failed continuity."),
        ("What are the exploratory values?", "UNVALIDATED: BR median 10.5/min (6.0–23.9); HR median 65.3/min (57.9–73.1), in 36 frames."),
    ]
    y = 1.42
    for i, (q, a) in enumerate(qa):
        fill = PANEL if i % 2 == 0 else PANEL2
        box(s, 0.62, y, 12.08, 0.78, fill=fill, line=BG, radius=True)
        text(s, q, 0.9, y + 0.17, 3.1, 0.35, size=14, color=ORANGE, bold=True)
        text(s, a, 4.15, y + 0.12, 8.2, 0.5, size=13, color=WHITE)
        y += 0.86
    box(s, 0.78, 6.72, 11.75, 0.3, fill=rgb("#173d4a"), line=TEAL, radius=True)
    text(s, "Bottom line: the method, thresholds, run outputs, assumptions, and limitations are reproducible; physical calibration and human-localization accuracy remain unverified.", 0.95, 6.8, 11.4, 0.15, size=10.5, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    source(s, "README.md; output/REPORT.md; output/summary.json; output/vital_summary.json")

    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
