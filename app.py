"""
Shelf Out-of-Stock Scanner
==========================
A Flask web app that accepts a photo of a retail shelf display,
uses Claude Vision to identify out-of-stock positions by detecting
shelf tags with no product behind them, and displays an annotated
image with red circles around empty positions.
"""

import os
import json
import base64
import uuid
from io import BytesIO
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from PIL import Image, ImageDraw, ImageFont
import anthropic

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = Path(__file__).parent / "uploads"
app.config["RESULTS_FOLDER"] = Path(__file__).parent / "results"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max

# Create directories
app.config["UPLOAD_FOLDER"].mkdir(exist_ok=True)
app.config["RESULTS_FOLDER"].mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def encode_image_base64(image_path):
    """Read an image file and return base64 encoded string."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def get_media_type(filename):
    """Get MIME type from filename."""
    ext = filename.rsplit(".", 1)[1].lower()
    return {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }[ext]


def analyze_shelf_image(image_path, filename):
    """
    Multi-pass analysis: upscale the image for clarity, then analyze full image
    plus per-shelf crops for detail, then verify and produce coordinates.
    """
    client = anthropic.Anthropic()

    media_type = get_media_type(filename)

    with Image.open(image_path) as img:
        orig_width, orig_height = img.size
        # Upscale small images so shelf tags are more visible
        if orig_width < 1200:
            scale = 1200 / orig_width
            new_size = (int(orig_width * scale), int(orig_height * scale))
            img_upscaled = img.resize(new_size, Image.LANCZOS)
        else:
            img_upscaled = img.copy()
            scale = 1.0
        width, height = img_upscaled.size

        # Save upscaled version temporarily
        upscaled_path = str(image_path) + "_upscaled.jpg"
        img_upscaled.save(upscaled_path, quality=95)

    image_data = encode_image_base64(upscaled_path)
    os.remove(upscaled_path)

    image_block = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": image_data,
        },
    }

    # ── Pass 1: Map all shelves and tags ──
    pass1_prompt = f"""You are analyzing a photo of a retail shelf display ({width} x {height} pixels).

IGNORE any annotations, arrows, circles, or text overlays in the image. Analyze only the actual fixture.

SHELF COUNTING: A shelf is any horizontal surface with a TAG STRIP (row of small labels) on its front lip. Count carefully — fixtures often have 6 or more shelves. The very bottom of the fixture may have a light panel (bright, uniform, NO tags on its lip) — that is NOT a shelf. But if a surface has tags on its lip, it IS a shelf even if it has few products on it.

TAG-FIRST METHOD — for each shelf (numbered 1 from top):
1. Look at the FRONT LIP of that shelf and count every shelf tag (small white/colored label).
2. Number the tags left to right: Tag 1, Tag 2, Tag 3, etc.
3. For EACH tag, look at the shelf surface directly BEHIND it from the camera's perspective:
   - If you see a product box/item rising up at the expected depth: STOCKED
   - If you see only bare flat shelf surface or the back wall with no product: EMPTY
4. To confirm EMPTY vs STOCKED: compare to a STOCKED neighbor on the same shelf. Behind a stocked tag you see a product rising up from the shelf. Behind an empty tag you see only flat shelf or back wall at the same viewing angle.

PERSPECTIVE NOTE: On lower shelves you see more shelf surface due to the downward camera angle. This is normal. The key distinction is: is there a PRODUCT rising up from the shelf behind this tag, or just flat surface? Compare left-right along the same shelf row.

Be precise. List every tag and its status (STOCKED or EMPTY) for every shelf."""

    pass1_response = client.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=4096,
        temperature=0,
        messages=[
            {"role": "user", "content": [image_block, {"type": "text", "text": pass1_prompt}]},
        ],
    )
    pass1_text = pass1_response.content[0].text

    # ── Pass 2: Verify with tag-first method ──
    pass2_prompt = """Now verify your analysis. Go back through EACH shelf and re-check using the tag-first method:

For each shelf, look at the front lip. Find each tag. Then look DIRECTLY BEHIND that tag from the camera's perspective. Do you see a product rising up, or just bare shelf?

COMMON MISTAKES TO AVOID:
- Miscounting shelves: Make sure you counted every shelf that has tags on its lip. The lowest shelf with tags is still a shelf, even if it has few products.
- Missing edge positions: Check the last 1-2 tags on the right edge of each shelf, and the first 1-2 tags on the left edge of each shelf.
- Confusing perspective for emptiness: On lower shelves, more bare surface is visible in front of products. That's normal. Focus on whether a product EXISTS behind the tag, not how much shelf surface you see.
- Confusing emptiness for perspective: If a tag has NO product behind it at all (just flat shelf or back wall) while neighboring tags on the same shelf DO have products, that is a genuine empty.

SPECIAL CHECK FOR THE LOWEST SHELF WITH TAGS (the bottom-most shelf before the light panel):
Look at this shelf very carefully. For each tag on the LEFT side of this shelf:
- Look DIRECTLY behind the tag. Do you see a product box/item, or just the bare shelf surface?
- Compare to the RIGHT side of the same shelf where products ARE present. The products there rise up from the shelf. Do you see the same thing behind the left-side tags?
- If the left-side tags have no product rising up behind them while the right-side tags do, those left positions are EMPTY.

List your revised tag-by-tag assessment for each shelf. Only change your previous assessment where you have clear reason to."""

    pass2_response = client.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=4096,
        temperature=0,
        messages=[
            {"role": "user", "content": [image_block, {"type": "text", "text": pass1_prompt}]},
            {"role": "assistant", "content": pass1_text},
            {"role": "user", "content": [image_block, {"type": "text", "text": pass2_prompt}]},
        ],
    )
    pass2_text = pass2_response.content[0].text

    # ── Pass 3: Reconcile and produce final coordinates ──
    pass3_prompt = f"""You performed two rounds of tag-by-tag analysis. Now reconcile them.

ROUND 1 FINDINGS:
{pass1_text}

ROUND 2 FINDINGS:
{pass2_text}

RECONCILIATION RULES:
- If a position was marked EMPTY in EITHER round and the other round did not explicitly mark it STOCKED with clear justification, include it as empty.
- If the tag count differs between rounds for a shelf, use the HIGHER count — it is easier to undercount tags than to hallucinate them.
- If one round found MORE empty positions on a shelf than the other, re-examine that shelf in the image to determine the correct count.

Place coordinates on the PRODUCT AREA (above/behind the tag), not on the tag itself.
Use stocked products on the same row to calibrate the correct y-coordinate.

The original image is {orig_width} x {orig_height} pixels. All coordinates must be in these dimensions.

Respond with ONLY valid JSON:
{{
  "image_width": {orig_width},
  "image_height": {orig_height},
  "total_shelves": <int>,
  "analysis_notes": "<brief summary>",
  "empty_positions": [
    {{
      "center_x": <int>,
      "center_y": <int>,
      "width": <int>,
      "height": <int>,
      "shelf_number": <int, from top>,
      "row": "front",
      "position_from_left": <int>,
      "tag_text": "<string or null>",
      "confidence": <float 0-1>
    }}
  ]
}}"""

    pass3_response = client.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=4096,
        temperature=0,
        messages=[
            {"role": "user", "content": [image_block, {"type": "text", "text": pass1_prompt}]},
            {"role": "assistant", "content": pass1_text},
            {"role": "user", "content": [image_block, {"type": "text", "text": pass2_prompt}]},
            {"role": "assistant", "content": pass2_text},
            {"role": "user", "content": [image_block, {"type": "text", "text": pass3_prompt}]},
        ],
    )

    response_text = pass3_response.content[0].text

    # Try to parse JSON, handling potential markdown code blocks
    json_text = response_text
    if "```json" in json_text:
        json_text = json_text.split("```json")[1].split("```")[0]
    elif "```" in json_text:
        json_text = json_text.split("```")[1].split("```")[0]

    return json.loads(json_text.strip())


def annotate_image(image_path, analysis_result):
    """
    Draw red circles on the image at each empty position.
    If tag_text is available and the image is high-res enough, add text labels.
    Returns the annotated PIL Image.
    """
    img = Image.open(image_path).copy()
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # Try to load a good font, fall back to default
    font = None
    font_small = None
    try:
        font_size = max(12, min(20, width // 50))
        font_small_size = max(10, min(16, width // 60))
        for font_path in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNSMono.ttf",
        ]:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                font_small = ImageFont.truetype(font_path, font_small_size)
                break
    except Exception:
        pass

    if font is None:
        font = ImageFont.load_default()
        font_small = font

    empty_positions = analysis_result.get("empty_positions", [])
    line_width = max(3, width // 150)

    for pos in empty_positions:
        cx = pos["center_x"]
        cy = pos["center_y"]
        w = pos["width"]
        h = pos["height"]
        confidence = pos.get("confidence", 0.5)
        tag_text = pos.get("tag_text")

        # Only draw positions with reasonable confidence
        if confidence < 0.4:
            continue

        # Draw red ellipse
        x1 = cx - w // 2
        y1 = cy - h // 2
        x2 = cx + w // 2
        y2 = cy + h // 2

        for i in range(line_width):
            draw.ellipse([x1 - i, y1 - i, x2 + i, y2 + i], outline="red")

        # If tag text is available and image is high-res, add label
        if tag_text and width >= 1000:
            # Draw text with background for readability
            text_x = x2 + 5
            text_y = cy - 8

            # Keep text within image bounds
            bbox = draw.textbbox((text_x, text_y), tag_text, font=font_small)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            if text_x + text_w > width - 5:
                text_x = x1 - text_w - 5

            # Background rectangle
            padding = 3
            draw.rectangle(
                [text_x - padding, text_y - padding,
                 text_x + text_w + padding, text_y + text_h + padding],
                fill="white",
                outline="red",
                width=1,
            )
            draw.text((text_x, text_y), tag_text, fill="red", font=font_small)

    # Draw legend
    legend_text = f"Out of Stock: {len([p for p in empty_positions if p.get('confidence', 0.5) >= 0.4])} position(s)"
    legend_bbox = draw.textbbox((0, 0), legend_text, font=font)
    lw = legend_bbox[2] - legend_bbox[0]
    lh = legend_bbox[3] - legend_bbox[1]

    padding = 6
    draw.rectangle(
        [8, 8, 8 + lw + 2 * padding, 8 + lh + 2 * padding],
        fill="white",
        outline="red",
        width=2,
    )
    draw.text((8 + padding, 8 + padding), legend_text, fill="red", font=font)

    return img


# ─── Routes ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Upload page with file chooser."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Handle file upload, analyze, annotate, and redirect to results."""
    if "file" not in request.files:
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return redirect(url_for("index"))

    # Save uploaded file with unique ID
    file_id = str(uuid.uuid4())[:8]
    ext = file.filename.rsplit(".", 1)[1].lower()
    original_name = file.filename
    saved_name = f"{file_id}.{ext}"
    upload_path = app.config["UPLOAD_FOLDER"] / saved_name
    file.save(upload_path)

    return render_template(
        "analyzing.html",
        file_id=file_id,
        ext=ext,
        original_name=original_name,
    )


@app.route("/analyze/<file_id>/<ext>/<original_name>")
def analyze(file_id, ext, original_name):
    """Run the analysis (called via AJAX from the analyzing page)."""
    saved_name = f"{file_id}.{ext}"
    upload_path = app.config["UPLOAD_FOLDER"] / saved_name

    try:
        # Analyze with Claude Vision
        analysis = analyze_shelf_image(str(upload_path), saved_name)

        # Save analysis JSON
        json_path = app.config["RESULTS_FOLDER"] / f"{file_id}.json"
        with open(json_path, "w") as f:
            json.dump(analysis, f, indent=2)

        # Annotate image
        annotated_img = annotate_image(str(upload_path), analysis)
        annotated_name = f"{file_id}_annotated.jpg"
        annotated_path = app.config["RESULTS_FOLDER"] / annotated_name
        annotated_img.save(str(annotated_path), quality=95)

        return jsonify({
            "success": True,
            "redirect": url_for("results", file_id=file_id, ext=ext, original_name=original_name),
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/results/<file_id>/<ext>/<original_name>")
def results(file_id, ext, original_name):
    """Display side-by-side original and annotated images."""
    # Load analysis JSON
    json_path = app.config["RESULTS_FOLDER"] / f"{file_id}.json"
    analysis = {}
    if json_path.exists():
        with open(json_path) as f:
            analysis = json.load(f)

    empty_count = len([
        p for p in analysis.get("empty_positions", [])
        if p.get("confidence", 0.5) >= 0.4
    ])

    return render_template(
        "results.html",
        file_id=file_id,
        ext=ext,
        original_name=original_name,
        analysis=analysis,
        empty_count=empty_count,
    )


@app.route("/uploads/<filename>")
def serve_upload(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/results_files/<filename>")
def serve_result(filename):
    return send_from_directory(app.config["RESULTS_FOLDER"], filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    print(f"Starting Shelf Out-of-Stock Scanner at http://localhost:{port}")
    app.run(debug=debug, host="0.0.0.0", port=port)
