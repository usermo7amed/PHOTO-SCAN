"""OpenCV prototype for Photo Unifier's processing routes.

The Android app chooses a route with ML Kit and a page finder. This standalone prototype accepts a
manual --mode because it does not embed ML Kit's model. The numbers here are the same as in the
Kotlin code (Enhancer.kt, ImageMetrics.kt, TonePlanner.kt, PageFinder.kt, LookMatcher.kt): tune
them here on real photos first, then copy them across.

Every photo route measures the image first and only applies the corrections the image actually
needs (see measure() and plan_for()). A photo can also be pulled towards a common look
(look_stats() and match_look()), so a batch shot under different light ends up consistent.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

MAX_SIDE = 4096
ANALYSIS_SIDE = 640

# neutral pixels (white balance)
NEUTRAL_CHROMA = 14.0          # lower bound of the neutral-pixel chroma limit
NEUTRAL_CHROMA_MAX = 35.0      # upper bound: the limit follows a strong cast but never reaches coloured objects
MIN_NEUTRAL_FRACTION = 0.05
WB_MIN, WB_MAX = 0.85, 1.2
WB_TRIGGER = 3.0               # Lab units of cast needed before white balance is touched

# when to touch what
BRIGHT_TRIGGER = 0.03          # share of pixels with L >= 235
DARK_TRIGGER = 0.05            # share of pixels with L <= 45
MEAN_DARK_TRIGGER = 90.0
CONTRAST_STD = 60.0            # L std below this adds local contrast
SAT_OK = 0.35                  # mean saturation at or above this needs no boost
VIBRANCE_GAIN = 1.6            # vibrance = (SAT_OK - saturation) * VIBRANCE_GAIN, up to the route's ceiling
VIBRANCE_MIN = 0.03            # anything weaker is not worth a pass over the image

SHADOW_LIFT = 0.12             # largest lift of pure black (share of full scale) at alpha_low = 1
SHADOW_LIFT_RANGE = 0.25       # lightness below which the lift acts

# per route: ceilings for the measured plan. shoulder/toe are the lightness above/below which the
# tone mapping starts to act, so a legitimately bright wall or a dark door is not flattened to grey.
ROUTES = {
    "interior": dict(shoulder=0.80, toe=0.40, high=0.50, low=0.55, gain_min=0.62, gain_max=2.25,
                     clahe=2.0, vibrance_max=0.35, sharp=0.24, white_balance=True),
    "exterior": dict(shoulder=0.72, toe=0.32, high=0.55, low=0.45, gain_min=0.55, gain_max=1.95,
                     clahe=1.8, vibrance_max=0.50, sharp=0.22, white_balance=False),
    "auto": dict(shoulder=0.78, toe=0.36, high=0.45, low=0.50, gain_min=0.65, gain_max=2.20,
                 clahe=2.0, vibrance_max=0.30, sharp=0.25, white_balance=True),
    "poster": dict(shoulder=0.85, toe=0.30, high=0.30, low=0.30, gain_min=0.78, gain_max=1.60,
                   clahe=1.45, vibrance_max=0.0, sharp=0.16, white_balance=False),
}

# page detection
DOC_MIN_AREA, DOC_MAX_AREA = 0.20, 0.97
DOC_MIN_RECTANGULARITY = 0.85
DOC_MIN_SUPPORT = 0.70         # share of the outline that must sit on a real edge
DOC_MIN_CONTRAST = 12.0        # mean gray difference between inside and outside
STRICT_MIN_AREA = 0.30         # without a label from ML Kit, only a very clear page is cut out
STRICT_MIN_RECTANGULARITY = 0.93
STRICT_MIN_SUPPORT = 0.85
BORDER_BAND = 0.06             # share of the frame used as the background sample
FG_MAHALANOBIS2 = 16.0         # squared distance (4 sigma) from the border colours
EDGE_TRIM = 0.006              # crop this share after the warp to drop table-edge residue
POSTER_COLORFUL = 0.15         # share of strongly coloured pixels that makes a page a poster

# paper shadows
PAPER_WHITE = 247.0            # lightness the paper level is brought to
PAPER_GAIN_MAX = 2.0
PAPER_SCALE = 8                # the paper level is estimated on a 1/8 copy
PAPER_PERCENTILE = 95          # the brightest pixels define "paper"
PAPER_FIRST_RATIO = 0.5        # first guess: pixels above this share of that level are paper
PAPER_REL_RATIO = 0.62         # then: pixels above this share of their own local level are paper
PAPER_PASSES = 2
PAPER_INK_GROW = 5             # max filter (px on the 1/8 copy) that removes thin text strokes
PAPER_SIGMA_DIV = 40.0         # blur sigma = longest side of the 1/8 copy / this
PAPER_WB_RATIO = 0.8           # paper white balance uses pixels at least this share of the paper level
DOCUMENT_CLAHE = 1.2
DOCUMENT_SHARP = 0.12

# unifying a batch
LOOK_L_SHIFT_MAX = 30.0        # largest lightness shift (0..255 scale) a look match may apply
LOOK_STD_MIN, LOOK_STD_MAX = 0.75, 1.35
LOOK_AB_SHIFT_MAX = 10.0


def clip(value, low, high):
    return max(low, min(high, value))


# ------------------------------------------------------------------ measuring
def analysis_copy(rgb):
    """Downscaled copy (longest side <= ANALYSIS_SIDE) and the scale applied."""
    h, w = rgb.shape[:2]
    scale = ANALYSIS_SIDE / max(h, w)
    if scale >= 1.0:
        return rgb.copy(), 1.0
    size = (max(round(w * scale), 1), max(round(h * scale), 1))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA), scale


def neutral_threshold(chroma, lightness):
    """Chroma limit for 'probably neutral' pixels: the quietest quarter of the midtones.

    A fixed limit misses strong casts (a warm lamp pushes even a white wall past it);
    the quartile follows the cast, and the cap keeps coloured objects out.
    """
    mid = (lightness > 40) & (lightness < 235)
    if int(mid.sum()) < 100:
        return NEUTRAL_CHROMA
    quartile = float(np.quantile(chroma[mid], 0.25)) + 1.0   # +1: the test below is strict
    return clip(quartile, NEUTRAL_CHROMA, NEUTRAL_CHROMA_MAX)


def measure(rgb_small):
    lab = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2LAB)
    lightness, a, b = cv2.split(lab)
    hsv = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2HSV)
    af = a.astype(np.float32) - 128.0
    bf = b.astype(np.float32) - 128.0
    chroma = np.hypot(af, bf)
    limit = neutral_threshold(chroma, lightness)
    neutral = (chroma < limit) & (lightness > 40) & (lightness < 235)
    metrics = {
        "mean": float(lightness.mean()),
        "std": float(lightness.std()),
        "bright": float((lightness >= 235).mean()),
        "dark": float((lightness <= 45).mean()),
        "saturation": float(hsv[..., 1].mean() / 255.0),
        "neutral": float(neutral.mean()),
        "neutral_chroma": limit,
        "cast": 0.0,
    }
    if metrics["neutral"] >= MIN_NEUTRAL_FRACTION:
        metrics["cast"] = float(np.hypot(af[neutral].mean(), bf[neutral].mean()))
    return metrics


def plan_for(mode, m):
    """Measured plan: strengths grow with the measured problem and stop at the route's ceiling."""
    route = ROUTES[mode]
    plan = dict(route)
    plan["alpha_high"] = 0.0 if m["bright"] < BRIGHT_TRIGGER else clip(0.2 + m["bright"], 0.2, route["high"])
    dark = m["dark"] > DARK_TRIGGER or m["mean"] < MEAN_DARK_TRIGGER
    plan["alpha_low"] = 0.0 if not dark else clip(
        0.15 + m["dark"] + max(0.0, 100.0 - m["mean"]) / 200.0, 0.15, route["low"])
    plan["clahe"] = 0.0 if m["std"] >= CONTRAST_STD else clip(
        0.8 + (CONTRAST_STD - m["std"]) / 10.0, 1.0, route["clahe"])
    vibrance = min(max(0.0, SAT_OK - m["saturation"]) * VIBRANCE_GAIN, route["vibrance_max"])
    plan["vibrance"] = vibrance if vibrance >= VIBRANCE_MIN else 0.0
    plan["white_balance"] = bool(route["white_balance"] and m["neutral"] >= MIN_NEUTRAL_FRACTION
                                 and m["cast"] >= WB_TRIGGER)
    plan["neutral_chroma"] = m["neutral_chroma"]
    return plan


def describe(plan):
    """Short Arabic labels for what the plan will do (same words as the app's log)."""
    steps = []
    if plan["white_balance"]:
        steps.append("ضبط توازن اللون")
    if plan["alpha_high"] > 0:
        steps.append("خفض السطوع العالي")
    if plan["alpha_low"] > 0:
        steps.append("رفع الظلال")
    if plan["clahe"] > 0:
        steps.append("تقوية التباين")
    if plan["vibrance"] > 0:
        steps.append("تحسين الألوان")
    return steps


# ------------------------------------------------------------------ enhancing
def gray_world(rgb, neutral_chroma=NEUTRAL_CHROMA):
    """Estimate white balance from near-neutral, midtone pixels only."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    _, a, b = cv2.split(lab)
    chroma = np.hypot(a.astype(np.float32) - 128.0, b.astype(np.float32) - 128.0)
    lightness = lab[:, :, 0]
    mask = ((chroma < neutral_chroma) & (lightness > 40) & (lightness < 235)).astype(np.uint8)
    if float(mask.mean()) < MIN_NEUTRAL_FRACTION:
        return rgb
    return _apply_white_balance(rgb, cv2.mean(rgb, mask)[:3])


def _apply_white_balance(rgb, means):
    gray = sum(means) / 3.0
    corrected = [cv2.convertScaleAbs(channel, alpha=float(np.clip(gray / max(mean, 1e-3), WB_MIN, WB_MAX)))
                 for channel, mean in zip(cv2.split(rgb), means)]
    return cv2.merge(corrected)


def local_tone_map(lightness, plan):
    """Pull regions brighter than the shoulder down by alpha_high, lift regions darker than the toe by alpha_low.

    The gain is worked out on a 1/8 copy and only then scaled up, which keeps the full-size work to one
    multiplication.
    """
    h, w = lightness.shape
    normalized = lightness.astype(np.float32) / 255.0
    small = cv2.resize(normalized, (max(w // 8, 1), max(h // 8, 1)), interpolation=cv2.INTER_AREA)
    illumination = cv2.GaussianBlur(small, (0, 0), max(small.shape) / 20.0) + 1e-3
    gain = np.ones_like(illumination)
    if plan["alpha_high"] > 0:
        gain *= np.minimum(np.power(plan["shoulder"] / illumination, plan["alpha_high"]), 1.0)
    if plan["alpha_low"] > 0:
        gain *= np.maximum(np.power(plan["toe"] / illumination, plan["alpha_low"]), 1.0)
    gain = np.clip(gain, plan["gain_min"], plan["gain_max"])
    gain = cv2.resize(gain, (w, h), interpolation=cv2.INTER_LINEAR)
    mapped = np.clip(normalized * gain, 0.0, 1.0)
    if plan["alpha_low"] > 0:
        # a gain cannot lift near-black; a small toe lift keeps deep shadows from being a flat black
        deep = np.clip(1.0 - mapped / SHADOW_LIFT_RANGE, 0.0, 1.0)
        mapped = mapped + plan["alpha_low"] * SHADOW_LIFT * deep * deep
    return (np.clip(mapped, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def vibrance_table(amount):
    """Lookup table for the HSV saturation channel: weakly coloured pixels gain most, vivid ones barely move."""
    s = np.arange(256, dtype=np.float32) / 255.0
    return np.clip(s * (1.0 + amount * (1.0 - s)) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def paper_level(small):
    """Smooth paper lightness over a page, from paper pixels only (normalized convolution).

    Thin text strokes are removed with a max filter. Ink and big dark blocks are then left out of the
    average, and are filled in from the paper around them. Pixels are judged against their *local* level
    in later passes, so the deep part of a shadow still counts as paper while ink does not.
    """
    small = cv2.dilate(small, np.ones((PAPER_INK_GROW, PAPER_INK_GROW), np.uint8))
    sigma = max(small.shape) / PAPER_SIGMA_DIV
    top = float(np.percentile(small, PAPER_PERCENTILE))
    mask = (small >= PAPER_FIRST_RATIO * top).astype(np.float32)
    level = None
    for _ in range(PAPER_PASSES + 1):
        weighted = cv2.GaussianBlur(small * mask, (0, 0), sigma)
        weight = cv2.GaussianBlur(mask, (0, 0), sigma)
        level = np.where(weight > 1e-3, weighted / np.maximum(weight, 1e-3), top).astype(np.float32)
        mask = (small >= PAPER_REL_RATIO * np.maximum(level, 1.0)).astype(np.float32)
    return level, top


def remove_paper_shadows(lightness):
    """Flatten shadows on a paper's L channel (not its colours) and bring the paper to PAPER_WHITE."""
    h, w = lightness.shape
    small = cv2.resize(lightness, (max(w // PAPER_SCALE, 1), max(h // PAPER_SCALE, 1)),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
    level, _ = paper_level(small)
    level = cv2.resize(level, (w, h), interpolation=cv2.INTER_LINEAR)
    gain = np.minimum(PAPER_WHITE / np.maximum(level, 1.0), PAPER_GAIN_MAX)
    return np.clip(lightness.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def paper_white_balance(rgb):
    """Make the paper itself neutral: average the colour of paper-bright pixels and balance on it."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness = lab[:, :, 0]
    top = float(np.percentile(lightness, PAPER_PERCENTILE))
    mask = (lightness >= PAPER_WB_RATIO * top).astype(np.uint8)
    if float(mask.mean()) < MIN_NEUTRAL_FRACTION:
        return rgb
    return _apply_white_balance(rgb, cv2.mean(rgb, mask)[:3])


def _apply_clahe(lightness, clip_limit):
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(lightness)


def _sharpen(rgb, strength):
    if strength <= 0:
        return rgb
    blur = cv2.GaussianBlur(rgb, (0, 0), 1.6)
    return cv2.addWeighted(rgb, 1.0 + strength, blur, -strength, 0)


# ------------------------------------------------------------------ one common look
def look_stats(rgb_small):
    """The few numbers that describe how a photo looks: lightness mean/spread and the average colour."""
    lab = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2LAB)
    lightness, a, b = cv2.split(lab)
    return {"l_mean": float(lightness.mean()), "l_std": float(lightness.std()),
            "a_mean": float(a.mean()) - 128.0, "b_mean": float(b.mean()) - 128.0}


def median_look(stats_list):
    """The middle look of a batch: the median of every number, so one odd photo does not drag the rest."""
    return {key: float(np.median([s[key] for s in stats_list])) for key in stats_list[0]}


def look_table(own, target, strength):
    """Lookup table that moves lightness part of the way (strength 0..1) from `own` towards `target`."""
    ratio = clip(target["l_std"] / max(own["l_std"], 1.0), LOOK_STD_MIN, LOOK_STD_MAX)
    scale = 1.0 + (ratio - 1.0) * strength
    shift = clip(target["l_mean"] - own["l_mean"], -LOOK_L_SHIFT_MAX, LOOK_L_SHIFT_MAX) * strength
    values = np.arange(256, dtype=np.float32)
    return np.clip((values - own["l_mean"]) * scale + own["l_mean"] + shift + 0.5, 0, 255).astype(np.uint8)


def match_look(rgb, target, strength):
    """Pull a finished photo towards a common look. strength 0 leaves it alone, 1 goes all the way."""
    if strength <= 0:
        return rgb
    small, _ = analysis_copy(rgb)
    own = look_stats(small)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness, a, b = cv2.split(lab)
    lightness = cv2.LUT(lightness, look_table(own, target, strength))
    shift_a = clip(target["a_mean"] - own["a_mean"], -LOOK_AB_SHIFT_MAX, LOOK_AB_SHIFT_MAX) * strength
    shift_b = clip(target["b_mean"] - own["b_mean"], -LOOK_AB_SHIFT_MAX, LOOK_AB_SHIFT_MAX) * strength
    a = cv2.convertScaleAbs(a, alpha=1.0, beta=shift_a)
    b = cv2.convertScaleAbs(b, alpha=1.0, beta=shift_b)
    return cv2.cvtColor(cv2.merge([lightness, a, b]), cv2.COLOR_LAB2RGB)


# ------------------------------------------------------------------ page finding
def order_points(points):
    """Corners ordered top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
                    dtype=np.float32)


def _quad_from_contour(contour):
    """Smallest epsilon that turns the convex hull into 4 corners."""
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    for eps in (0.015, 0.02, 0.03, 0.04, 0.06):
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == 4:
            return order_points(approx)
        if len(approx) < 4:
            return None
    return None


def _candidate_quads(mask, image_area, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        quad = _quad_from_contour(contour)
        if quad is None:
            continue
        fraction = cv2.contourArea(quad) / image_area
        if min_area <= fraction <= DOC_MAX_AREA:
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            yield quad, fraction, min(hull_area / max(cv2.contourArea(quad), 1.0), 1.0)


def edge_support(quad, edges_near):
    """Share of points along the quad outline that lie on a real image edge."""
    hits = total = 0
    h, w = edges_near.shape
    for i in range(4):
        p0, p1 = quad[i], quad[(i + 1) % 4]
        for t in np.linspace(0.05, 0.95, 40):
            x, y = p0 + (p1 - p0) * t
            xi, yi = int(round(float(x))), int(round(float(y)))
            total += 1
            if 0 <= xi < w and 0 <= yi < h and edges_near[yi, xi] > 0:
                hits += 1
    return hits / max(total, 1)


def foreground_mask(rgb_small):
    """Pixels that do not look like the frame border (a page is usually surrounded by table or wall).

    The border colours are modelled as one Gaussian in Lab; everything far from it is foreground.
    A shadow on the page does not matter: it stays far from the table or wall colour.
    """
    lab = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    band = max(int(BORDER_BAND * min(h, w)), 4)
    border = np.zeros((h, w), bool)
    border[:band, :] = border[-band:, :] = True
    border[:, :band] = border[:, -band:] = True
    samples = lab[border]
    mean = samples.mean(axis=0)
    inverse = np.linalg.inv(np.cov(samples.T) + np.eye(3, dtype=np.float32) * 4.0)
    d = (lab - mean).reshape(-1, 3)
    distance2 = np.einsum("ij,jk,ik->i", d, inverse, d).reshape(h, w)
    fg = (distance2 > FG_MAHALANOBIS2).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))


def find_quad(rgb_small, strict=False):
    """Four corners (tl, tr, br, bl) of a page in the coordinates of rgb_small, or None.

    strict asks for a bigger, squarer and better-outlined page. It is what lets the app cut out a page
    without any hint from the photo's labels.
    """
    min_area = STRICT_MIN_AREA if strict else DOC_MIN_AREA
    min_rect = STRICT_MIN_RECTANGULARITY if strict else DOC_MIN_RECTANGULARITY
    min_support = STRICT_MIN_SUPPORT if strict else DOC_MIN_SUPPORT

    gray = cv2.GaussianBlur(cv2.cvtColor(rgb_small, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    h, w = gray.shape
    image_area = float(h * w)

    median = float(np.median(gray))
    raw = cv2.Canny(gray, max(0.66 * median, 20), max(1.33 * median, 60))
    # colour edges too: an orange poster on a grey wall can match the wall's brightness
    lab = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2LAB)
    for channel in (lab[..., 1], lab[..., 2]):
        raw = cv2.bitwise_or(raw, cv2.Canny(cv2.GaussianBlur(channel, (5, 5), 0), 15, 45))
    edges_near = cv2.dilate(raw, np.ones((7, 7), np.uint8))
    edges_closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    best, best_score = None, 0.0
    for mask in (foreground_mask(rgb_small), edges_closed, threshold, 255 - threshold):
        for quad, fraction, rectangularity in _candidate_quads(mask, image_area, min_area):
            if rectangularity < min_rect:
                continue
            margin = 0.03   # the whole frame is not a page
            if all((x < margin * w or x > (1 - margin) * w) and (y < margin * h or y > (1 - margin) * h)
                   for x, y in quad):
                continue
            tl, tr, br, bl = quad
            width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
            height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
            if height < 1 or not (0.3 <= width / height <= 3.5):
                continue
            support = edge_support(quad, edges_near)
            if support < min_support:
                continue
            if fraction < 0.9:
                inside = np.zeros_like(gray)
                cv2.fillConvexPoly(inside, quad.astype(np.int32), 255)
                if abs(float(gray[inside > 0].mean()) - float(gray[inside == 0].mean())) < DOC_MIN_CONTRAST:
                    continue
            score = fraction * rectangularity * support
            if score > best_score:
                best, best_score = quad, score
    return best


def warp_page(bgr, quad_small, scale):
    corners = (quad_small / scale).astype(np.float32)
    tl, tr, br, bl = corners
    width = int(np.clip(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))), 2, MAX_SIDE))
    height = int(np.clip(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))), 2, MAX_SIDE))
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(corners, target)
    page = cv2.warpPerspective(bgr, transform, (width, height), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
    trim = int(round(EDGE_TRIM * min(width, height)))
    if trim > 0 and width - 2 * trim >= 64 and height - 2 * trim >= 64:
        page = page[trim:height - trim, trim:width - trim]
    return page


def rectify_page(bgr, strict=False):
    """Rectify the page or poster in the frame; otherwise keep the input."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    small, scale = analysis_copy(rgb)
    quad = find_quad(small, strict)
    if quad is None:
        return bgr
    page = warp_page(bgr, quad, scale)
    return bgr if min(page.shape[:2]) < 80 else page


def colorful_share(rgb_small):
    """Share of strongly coloured pixels: high for a poster, low for a printed page."""
    hsv = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2HSV)
    return float(((hsv[..., 1] > 70) & (hsv[..., 2] > 70)).mean())


def detect_mode(bgr):
    """Guess the route from the picture alone: a clear page is a document or a poster, anything else a photo."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    small, scale = analysis_copy(rgb)
    quad = find_quad(small, strict=True)
    if quad is None:
        return "auto"
    page = warp_page(rgb, quad, scale)
    page_small, _ = analysis_copy(page)
    return "poster" if colorful_share(page_small) >= POSTER_COLORFUL else "document"


# ------------------------------------------------------------------ routes
def enhance(bgr, mode="auto", rectify=True, look=None, look_strength=0.0):
    """Enhance one image using its selected route; auto is the conservative general route.

    look/look_strength pull a finished photo towards a common look (photos only: a page keeps its
    paper white and a poster its colours).
    """
    if mode not in {"document", *ROUTES.keys()}:
        raise ValueError(f"Unsupported mode: {mode}")

    source = rectify_page(bgr) if rectify and mode in {"document", "poster"} else bgr.copy()
    rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

    if mode == "document":
        rgb = paper_white_balance(rgb)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        lightness, a, b = cv2.split(lab)
        lightness = remove_paper_shadows(lightness)
        lightness = _apply_clahe(lightness, DOCUMENT_CLAHE)
        rgb = cv2.cvtColor(cv2.merge([lightness, a, b]), cv2.COLOR_LAB2RGB)
        return cv2.cvtColor(_sharpen(rgb, DOCUMENT_SHARP), cv2.COLOR_RGB2BGR)

    small, _ = analysis_copy(rgb)
    plan = plan_for(mode, measure(small))
    if plan["white_balance"]:
        rgb = gray_world(rgb, plan["neutral_chroma"])
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness, a, b = cv2.split(lab)
    if plan["alpha_high"] > 0 or plan["alpha_low"] > 0:
        lightness = local_tone_map(lightness, plan)
    if plan["clahe"] > 0:
        lightness = _apply_clahe(lightness, plan["clahe"])
    rgb = cv2.cvtColor(cv2.merge([lightness, a, b]), cv2.COLOR_LAB2RGB)
    if plan["vibrance"] > 0:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        rgb = cv2.cvtColor(cv2.merge([h, cv2.LUT(s, vibrance_table(plan["vibrance"])), v]), cv2.COLOR_HSV2RGB)
    rgb = _sharpen(rgb, plan["sharp"])
    if look is not None and mode != "poster":
        rgb = match_look(rgb, look, look_strength)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def stats(name, bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    print(f"{name}: mean={gray.mean():.1f} std={gray.std():.1f} "
          f"burnt(>=250)={100 * (gray >= 250).mean():.1f}% "
          f"dark(<=30)={100 * (gray <= 30).mean():.1f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=["auto", "interior", "exterior", "document", "poster", "detect"],
                        default="auto", help="detect picks document/poster from a clearly visible page")
    args = parser.parse_args()

    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"Could not read input image: {args.input}")
    mode = detect_mode(image) if args.mode == "detect" else args.mode
    result = enhance(image, mode)
    print("mode  :", mode)
    stats("before", image)
    stats("after ", result)
    if mode in ROUTES:
        small, _ = analysis_copy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        print("plan  :", ", ".join(describe(plan_for(mode, measure(small)))) or "no change needed")
    height = 520
    pair = [cv2.resize(img, (round(img.shape[1] * height / img.shape[0]), height), interpolation=cv2.INTER_AREA)
            for img in (image, result)]
    if not cv2.imwrite(str(args.output), np.hstack(pair), [cv2.IMWRITE_JPEG_QUALITY, 90]):
        parser.error(f"Could not write output image: {args.output}")


if __name__ == "__main__":
    main()
