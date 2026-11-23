# ===== Block 1: Import libraries =====
import time
import random
import shutil
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    ssd300_vgg16, SSD300_VGG16_Weights
)
from PIL import Image, ImageOps
from google.colab import files

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(42)
print('Device:', device)


# ===== Block 2: Load the pretrained Faster R-CNN model =====
weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
model = fasterrcnn_resnet50_fpn(weights=weights).to(device).eval()

# COCO class labels used by the pretrained model. The model returns COCO category ids,
# so this list is indexed directly by them ('N/A' marks ids that COCO does not use).
COCO_CLASSES = weights.meta['categories']
print('Object classes:', sum(c not in ('N/A', '__background__') for c in COCO_CLASSES))
print(COCO_CLASSES[1:12])


# ===== Block 3: Function for performing object detection =====
def detect_objects(image_path, model, threshold=0.5):
    print(f"Loading image from: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}. Please check the file path.")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float().div(255).to(device)

    with torch.no_grad():
        detections = model([input_tensor])[0]

    boxes = detections['boxes'].cpu().numpy()          # [xmin, ymin, xmax, ymax] in pixels
    class_indices = detections['labels'].cpu().numpy()
    scores = detections['scores'].cpu().numpy()

    valid_detections = scores >= threshold
    boxes = boxes[valid_detections]
    class_indices = class_indices[valid_detections]
    scores = scores[valid_detections]

    return image_rgb, boxes, class_indices, scores


# ===== Block 4: Function for displaying the image with bounding boxes =====
def display_image_with_detections(image, boxes, class_indices, scores, threshold=0.5, title=None):
    image = image.copy()
    scale = max(image.shape[:2]) / 1000  # thicker lines and text for large photos
    thickness = max(2, int(2 * scale))
    font_size = max(0.5, 0.7 * scale)

    for i, box in enumerate(boxes):
        if scores[i] >= threshold:
            left, top, right, bottom = box.astype(int)
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), thickness)
            label = f"{COCO_CLASSES[class_indices[i]]}: {scores[i]:.2f}"
            cv2.putText(image, label, (left, max(top - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                        font_size, (0, 255, 0), thickness)

    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()


# ===== Block 5: Task 1 - Test with your own image =====
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

try:
    image_rgb, detected_boxes, detected_classes, detected_scores = detect_objects(image_path, model, threshold=0.5)
    print(f"{len(detected_boxes)} objects detected:")
    for c, s in zip(detected_classes, detected_scores):
        print(f"  {COCO_CLASSES[c]}: {s:.2f}")
    display_image_with_detections(image_rgb, detected_boxes, detected_classes, detected_scores)
except FileNotFoundError as e:
    print(e)


# ===== Block 6: Task 2 - Two more models with different backbones =====
detectors = {
    'Faster R-CNN (ResNet50-FPN)': model,
    'Faster R-CNN (MobileNetV3-FPN)': fasterrcnn_mobilenet_v3_large_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1),
    'SSD300 (VGG16)': ssd300_vgg16(weights=SSD300_VGG16_Weights.COCO_V1),
}
for m in detectors.values():
    m.to(device).eval()

input_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float().div(255).to(device)

comparison = []
for name, m in detectors.items():
    with torch.no_grad():
        m([input_tensor])  # warm-up
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(5):
            out = m([input_tensor])[0]
        if device.type == 'cuda':
            torch.cuda.synchronize()
    t = (time.time() - start) / 5 * 1000

    keep = out['scores'] >= 0.5
    boxes = out['boxes'][keep].cpu().numpy()
    labels = out['labels'][keep].cpu().numpy()
    scores = out['scores'][keep].cpu().numpy()
    params = sum(p.numel() for p in m.parameters())
    comparison.append((name, params, len(boxes), t))

    print(f"\n{name}: {len(boxes)} objects detected")
    for c, s in zip(labels, scores):
        print(f"  {COCO_CLASSES[c]}: {s:.2f}")
    display_image_with_detections(image_rgb, boxes, labels, scores, title=name)

print(f"\n{'Model':32} {'Parameters':>12} {'Detections':>11} {'Time (ms)':>10}")
for name, params, n, t in comparison:
    print(f"{name:32} {params:12,} {n:11} {t:10.1f}")


# ===== Block 7a: Task 3 - Read the boxes drawn on the images =====
# Reads the boxes back from images that have labelled boxes drawn on them
# (e.g. Annotated_Images.zip): every class was drawn in its own exact colour with a name tag,
# so each tag is found by exact template matching and its rectangle is traced from the line pixels.
# The drawings are then painted out (inpainting) so the detectors are tested on clean pictures.
# Colours used when the boxes were drawn (RGB), one per class
CLASS_COLORS = {
    'trafficlight': (255, 193, 7),
    'stop': (220, 53, 69),
    'speedlimit': (0, 123, 255),
    'crosswalk': (40, 167, 69),
}


def draw_style(shape):
    h, w = shape[:2]
    return max(2, round(min(h, w) / 150)), max(0.4, min(h, w) / 600)


def tag_size(name, shape):
    _, fs = draw_style(shape)
    (tw, th), base = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    return tw, th, base


def draw_box(canvas, name, x1, y1, x2, y2, color, outline=True, tag=True):
    # Same drawing routine that produced the annotated images
    t, fs = draw_style(canvas.shape)
    tw, th, base = tag_size(name, canvas.shape)
    if outline:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, t)
    if tag:
        ty = max(y1, th + base + 4)
        cv2.rectangle(canvas, (x1, ty - th - base - 4), (x1 + tw + 6, ty), color, -1)
        cv2.putText(canvas, name, (x1 + 3, ty - base - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, (255, 255, 255), 1, cv2.LINE_AA)


def tag_template(name, color, shape):
    tw, th, base = tag_size(name, shape)
    th_tag = th + base + 4
    tmpl = np.zeros(shape, np.uint8)
    draw_box(tmpl, name, 0, th_tag, 1, th_tag + 1, color, outline=False)
    return (tmpl[:th_tag + 1, :tw + 7] == np.array(color, np.uint8)).all(-1).astype(np.float32), th_tag


def find_all_tags(img):
    # Locate every label tag by matching the exact rendered tag. Tags already found (and other
    # classes' drawings) are treated as wildcards, so a tag partly hidden under another is still found.
    h, w = img.shape[:2]
    masks = {n: (img == np.array(c, np.uint8)).all(-1) for n, c in CLASS_COLORS.items()}
    any_col = np.zeros((h, w), bool)
    for m in masks.values():
        any_col |= m
    found = []
    W = np.zeros((h, w), np.float32)  # wildcard pixels: covered by a tag found earlier
    while True:
        best = None
        for name, color in CLASS_COLORS.items():
            if not masks[name].any():
                continue
            F, th_tag = tag_template(name, color, img.shape)
            th_, tw_ = F.shape
            E = masks[name].astype(np.float32)
            wild = np.maximum(W, (any_col & ~masks[name]).astype(np.float32))
            P = lambda a: cv2.copyMakeBorder(a, 0, th_, 0, tw_, cv2.BORDER_CONSTANT, value=0)
            K = P((1 - wild))            # pixels inside the image that are not wildcards
            Ek = P(E * (1 - wild))
            agree = cv2.matchTemplate(Ek, F, cv2.TM_CCORR) + cv2.matchTemplate(K - Ek, 1 - F, cv2.TM_CCORR)
            known = cv2.matchTemplate(K, np.ones_like(F), cv2.TM_CCORR)
            ratio = np.where(known >= 0.25 * F.size, agree / np.maximum(known, 1), 0)
            y, x = np.unravel_index(np.argmax(ratio), ratio.shape)
            if ratio[y, x] >= 0.92 and (best is None or known[y, x] > best[0]):
                best = (known[y, x], name, int(x), int(y), th_tag, tw_, th_)
        if best is None:
            break
        _, name, x, y, th_tag, tw_, th_ = best
        found.append((name, x, y + th_tag))  # (class, x1, bottom row of the tag)
        W[y:y + th_, x:x + tw_] = 1
    return found


def run_end(line, start):
    # last index of the run of True values that begins at `start`
    i = start
    while i + 1 < len(line) and line[i + 1]:
        i += 1
    return i if line[start] else None


def best_box(mask, covered, free, name, color, shape, x1, y1_options, x2s, y2s, min_y2):
    # Smallest rectangle whose drawn outline is fully supported by the image
    h, w = shape[:2]
    best = None
    for y1 in y1_options:
        cands = sorted(((x2, y2) for x2 in set(x2s) for y2 in set(y2s)
                        if x2 > x1 + 2 and y2 > max(y1 + 2, min_y2)),
                       key=lambda b: (b[0] - x1) * (b[1] - y1))
        for x2, y2 in cands:
            if best is not None and (x2 - x1) * (y2 - y1) >= best[0]:
                break
            canvas = np.zeros(shape, np.uint8)
            draw_box(canvas, name, x1, y1, x2, y2, color, tag=False)
            out = (canvas == np.array(color, np.uint8)).all(-1)
            n = np.count_nonzero(out)
            if n == 0:
                continue
            if np.count_nonzero(out & mask & free) < 0.3 * n:   # a good part must be visibly drawn in this colour
                continue
            if np.count_nonzero(out & covered) >= 0.95 * n:
                best = ((x2 - x1) * (y2 - y1), (x1, y1, min(x2, w - 1), min(y2, h - 1)))
                break
    return best[1] if best else None


def edge_candidates(covered, mask, other_tags, x1, y1, t, shape):
    h, w = shape[:2]
    k = max(4, 2 * t)
    r0, r1 = min(h - 1, y1 + t + 1), min(h, y1 + t + 1 + k)
    c0, c1 = min(w - 1, x1 + t + 1), min(w, x1 + t + 1 + k)
    vert = [covered[r0:r1, c].all() for c in range(w)]
    horz = [covered[r, c0:c1].all() for r in range(h)]
    x2s = [c for c in range(x1 + 3, w) if vert[c] and (c + 1 >= w or not vert[c + 1])]
    y2s = [r for r in range(y1 + 3, h) if horz[r] and (r + 1 >= h or not horz[r + 1])]
    x2s = [c - d for c in x2s for d in range(t + 1)] + [w + 50]
    y2s = [r - d for r in y2s for d in range(t + 1)] + [h + 50]
    for m in (mask, covered):  # where the top edge and the left edge stop
        e = run_end(m[min(h - 1, y1), :], x1) if 0 <= y1 < h else None
        if e is not None:
            x2s += [e - d for d in range(t + 1)]
        e = run_end(m[:, x1], max(0, y1))
        if e is not None:
            y2s += [e - d for d in range(t + 1)]
    # edges hidden under other tags
    ys, xs = np.nonzero(other_tags[:, x1:x1 + 1])
    y2s += [int(y) for y in ys if y > y1 + 2]
    if 0 <= y1 < h:
        xs = np.nonzero(other_tags[y1])[0]
        x2s += [int(x) for x in xs if x > x1 + 2]
    return x2s, y2s


def fit_from_tag(img, mask, covered, free, other_tags, name, color, x1, ty, own_tag):
    t, _ = draw_style(img.shape)
    tw, th, base = tag_size(name, img.shape)
    th_tag = th + base + 4
    y1_options = [ty] if ty > th_tag else list(range(0, th_tag + 1))
    best = None
    for y1 in y1_options:
        x2s, y2s = edge_candidates(covered | own_tag, mask, other_tags, x1, y1, t, img.shape)
        box = best_box(mask, covered | own_tag, free, name, color, img.shape,
                       x1, [y1], x2s, y2s, ty + t)
        if box and (best is None or (box[2] - box[0]) * (box[3] - box[1]) < best[0]):
            best = ((box[2] - box[0]) * (box[3] - box[1]), box)
    return best[1] if best else None


def fit_untagged(img, comp, covered, free, other_tags, name, color):
    # Box whose tag is hidden under another tag: start from the lowest visible edge
    h, w = img.shape[:2]
    t, _ = draw_style(img.shape)
    ys, xs = np.nonzero(comp)
    y_bot = ys.max()
    row = xs[ys == y_bot]
    x1, x2 = int(row.min()), int(row.max())
    y2 = int(min(y_bot, h - 1) - t // 2) if y_bot < h - 1 else h + 50
    return best_box(comp, covered, free, name, color, img.shape, x1,
                    range(max(0, int(ys.min()) - 2 * t), y2 - 2), [x2, x2 + 1, x2 - 1], [y2], 0)


def refine(box, mask, covered, tags_area, name, color, shape, move_x1=False, move_y1=False):
    # Nudge the box edges by a pixel or two to best match the drawn line pixels
    h, w = shape[:2]
    t, _ = draw_style(shape)
    x1, y1, x2, y2 = box
    r0, r1 = max(0, y1 - 3 * t), min(h, y2 + 3 * t + 1)
    c0, c1 = max(0, x1 - 3 * t), min(w, x2 + 3 * t + 1)
    target = mask[r0:r1, c0:c1] & ~tags_area[r0:r1, c0:c1]
    cov = covered[r0:r1, c0:c1]
    R = range(-t, t + 1)
    best = None
    for a in (R if move_x1 else [0]):
        for b in (R if move_y1 else [0]):
            for c in R:
                for d in R:
                    bx = (x1 + a, y1 + b, x2 + c, y2 + d)
                    if bx[2] <= bx[0] + 2 or bx[3] <= bx[1] + 2:
                        continue
                    canvas = np.zeros(shape, np.uint8)
                    draw_box(canvas, name, *bx, color, tag=False)
                    out = (canvas == np.array(color, np.uint8)).all(-1)[r0:r1, c0:c1]
                    score = (np.count_nonzero(out & target) - 3 * np.count_nonzero(out & ~cov)
                             - np.count_nonzero(target & ~out))
                    if best is None or score > best[0]:
                        best = (score, bx)
    x1, y1, x2, y2 = best[1]
    return (max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2))


def recover_boxes(img):
    # img: RGB image with boxes drawn on it -> list of (class name, (xmin, ymin, xmax, ymax))
    h, w = img.shape[:2]
    masks = {n: (img == np.array(c, np.uint8)).all(-1) for n, c in CLASS_COLORS.items()}
    tags = find_all_tags(img)
    all_tags = np.zeros((h, w), bool)
    regions = []
    for name, x1, ty in tags:
        tw, th, base = tag_size(name, img.shape)
        r = np.zeros((h, w), bool)
        r[max(0, ty - th - base - 4):ty + 1, x1:x1 + tw + 7] = True
        regions.append(r)
        all_tags |= r
    objects = []
    explained = all_tags.copy()
    for (name, x1, ty), own in zip(tags, regions):
        color = CLASS_COLORS[name]
        other = all_tags & ~own
        for n2, m2 in masks.items():
            if n2 != name:
                other |= m2
        box = fit_from_tag(img, masks[name], masks[name] | other, ~other, all_tags & ~own,
                           name, color, x1, ty, own)
        if box:
            tw, th, base = tag_size(name, img.shape)
            box = refine(box, masks[name], masks[name] | other | own, all_tags, name, color, img.shape,
                         move_y1=ty <= th + base + 4)
            objects.append((name, box))
            canvas = np.zeros(img.shape, np.uint8)
            draw_box(canvas, name, *box, color, tag=False)
            explained |= (canvas == np.array(color, np.uint8)).all(-1)
    # boxes whose tag was hidden: colour pixels that no box explains yet
    explained = cv2.dilate(explained.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    for name, color in CLASS_COLORS.items():
        left = masks[name] & ~explained
        n, labels, stats, _ = cv2.connectedComponentsWithStats(left.astype(np.uint8), connectivity=8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 40:
                continue
            other = all_tags.copy()
            for n2, m2 in masks.items():
                if n2 != name:
                    other |= m2
            box = fit_untagged(img, labels == i, masks[name] | other | explained, ~other, all_tags, name, color)
            if box:
                box = refine(box, masks[name], masks[name] | other | explained, all_tags, name, color,
                             img.shape, move_x1=True, move_y1=True)
                objects.append((name, box))
    return objects


def clean_image(img, objects):
    # Paint over the drawn boxes and tags so the detectors see the original picture
    h, w = img.shape[:2]
    any_col = np.zeros((h, w), bool)
    for c in CLASS_COLORS.values():
        any_col |= (img == np.array(c, np.uint8)).all(-1)
    footprint = np.zeros((h, w), np.uint8)
    for name, box in objects:
        canvas = np.zeros(img.shape, np.uint8)
        draw_box(canvas, name, *box, CLASS_COLORS[name])
        footprint |= canvas.any(-1).astype(np.uint8)
    footprint = cv2.dilate(footprint | any_col.astype(np.uint8), np.ones((3, 3), np.uint8))
    return cv2.inpaint(img, footprint, 3, cv2.INPAINT_TELEA)


# ===== Block 7b: Task 3 - Upload and read the labelled evaluation data =====
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
MAX_IMAGES = 150   # how many labelled images to use (None = all)
SEED = 42          # same seed = same images every run


def open_image(path):
    # exif_transpose rotates phone photos the same way labelling tools display them
    return ImageOps.exif_transpose(Image.open(path)).convert('RGB')


def read_yolo_names(root):
    # Class names from classes.txt / obj.names / data.yaml (Roboflow, LabelImg, makesense)
    for txt in list(root.rglob('classes.txt')) + list(root.rglob('*.names')):
        names = [line.strip() for line in open(txt) if line.strip()]
        if names:
            return names
    for yml in list(root.rglob('data.yaml')) + list(root.rglob('*.yaml')):
        try:
            names = yaml.safe_load(open(yml)).get('names')
        except Exception:
            continue
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        if names:
            return [str(n) for n in names]
    return []


def pick(paths):
    # random but repeatable subset
    paths = sorted(paths)
    random.Random(SEED).shuffle(paths)
    return paths


def load_labelled_zip(zip_path, extract_dir):
    shutil.rmtree(extract_dir, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)
    root = Path(extract_dir)

    def usable(p):
        return '__MACOSX' not in p.parts and not p.name.startswith('.')

    images = {p.stem: p for p in root.rglob('*') if p.suffix.lower() in IMAGE_EXTS and usable(p)}
    xml_files = [p for p in root.rglob('*.xml') if usable(p)]
    txt_files = [p for p in root.rglob('*.txt') if usable(p) and p.name not in ('classes.txt', 'README.txt')]
    samples, skipped = [], 0
    enough = lambda: MAX_IMAGES is not None and len(samples) >= MAX_IMAGES

    if xml_files:
        label_format = 'Pascal VOC (.xml)'
        for xml_path in pick(xml_files):
            if enough():
                break
            ann = ET.parse(xml_path).getroot()
            img_path = images.get(xml_path.stem) or images.get(Path(ann.findtext('filename', '')).stem)
            if img_path is None:
                skipped += 1
                continue
            width, height = open_image(img_path).size
            xml_w = float(ann.findtext('size/width') or 0) or width
            xml_h = float(ann.findtext('size/height') or 0) or height
            sx, sy = width / xml_w, height / xml_h  # in case the image was resized after labelling
            objects = []
            for obj in ann.iter('object'):
                bb = obj.find('bndbox')
                x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ('xmin', 'ymin', 'xmax', 'ymax'))
                box = [max(0, x1 * sx), max(0, y1 * sy), min(width, x2 * sx), min(height, y2 * sy)]
                if box[2] - box[0] > 1 and box[3] - box[1] > 1:
                    objects.append((obj.findtext('name').strip(), box))
            samples.append((img_path, objects))
    elif txt_files:
        label_format = 'YOLO (.txt)'
        names = read_yolo_names(root)
        for txt_path in pick(txt_files):
            if enough():
                break
            img_path = images.get(txt_path.stem)
            if img_path is None:
                skipped += 1
                continue
            width, height = open_image(img_path).size
            objects = []
            for line in open(txt_path):
                values = line.split()
                if len(values) < 5:
                    continue
                cls = int(float(values[0]))
                xc, yc, bw, bh = (float(v) for v in values[1:5])
                box = [max(0, (xc - bw / 2) * width), max(0, (yc - bh / 2) * height),
                       min(width, (xc + bw / 2) * width), min(height, (yc + bh / 2) * height)]
                if box[2] - box[0] > 1 and box[3] - box[1] > 1:
                    name = names[cls] if cls < len(names) else f'class_{cls}'
                    objects.append((name.strip(), box))
            samples.append((img_path, objects))
    elif images:
        # No label files: read the boxes from the colours drawn on the images
        label_format = 'boxes drawn on the images (read from their colours)'
        clean_dir = root / '_clean'
        clean_dir.mkdir(exist_ok=True)
        todo = pick(list(images.values()))
        print(f'Reading the drawn boxes from {len(todo)} images...')
        for k, img_path in enumerate(todo):
            if enough():
                break
            img = np.array(open_image(img_path))
            objects = recover_boxes(img)
            if not objects:
                skipped += 1
                continue
            clean_path = clean_dir / (img_path.stem + '.png')
            Image.fromarray(clean_image(img, objects)).save(clean_path)
            samples.append((clean_path, [(name, [float(v) for v in box]) for name, box in objects]))
            if (k + 1) % 25 == 0:
                print(f'  {k + 1} images done')
    else:
        raise ValueError('No images or label files found in the zip.')

    if not samples:
        raise ValueError('No labelled images found. Use a zip with images + .xml/.txt label files, or '
                         'PNG images with the boxes drawn by visualize_annotations.py.')

    print('Label format:', label_format)
    print('Labelled images used:', len(samples))
    if skipped:
        print('Images skipped (no labels found):', skipped)
    counts = {}
    for _, objects in samples:
        for name, _ in objects:
            counts[name] = counts.get(name, 0) + 1
    for name, n in sorted(counts.items()):
        print(f'  {name}: {n} objects')
    return samples


# Upload Annotated_Images.zip (or any zip with Pascal VOC .xml / YOLO .txt labels)
uploaded = files.upload()
zip_name = next(name for name in uploaded if name.lower().endswith('.zip'))
samples = load_labelled_zip(zip_name, '/content/eval_data')


# ===== Block 7c: Task 3 - Check the labels =====
# Check that the labels were read correctly: ground-truth boxes on 6 of the evaluation images
label_colors = {}
palette = ['lime', 'red', 'deepskyblue', 'orange', 'magenta', 'yellow']
plt.figure(figsize=(18, 10))
for k, (path, objects) in enumerate(random.Random(1).sample(samples, min(6, len(samples)))):
    ax = plt.subplot(2, 3, k + 1)
    ax.imshow(open_image(path))
    for name, (x1, y1, x2, y2) in objects:
        color = label_colors.setdefault(name, palette[len(label_colors) % len(palette)])
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2))
        ax.text(x1, y1 - 3, name, color='black', fontsize=9, backgroundcolor=color)
    ax.set_title(Path(path).name)
    ax.axis('off')
plt.suptitle('Ground-truth labels used for evaluation')
plt.tight_layout()
plt.show()


# ===== Block 8: Task 3 - Match the labels to COCO classes =====
# The pretrained models only know the 80 COCO classes, so every label has to be matched to one.
# Road sign labels: 'trafficlight' and 'stop' are COCO classes under different names.
# 'speedlimit' and 'crosswalk' are not COCO classes, so the pretrained models cannot be scored on them.
CLASS_MAP = {'trafficlight': 'traffic light', 'stop': 'stop sign'}

ALIASES = {'mug': 'cup', 'phone': 'cell phone', 'mobile': 'cell phone', 'mobile phone': 'cell phone',
           'smartphone': 'cell phone', 'bike': 'bicycle', 'motorbike': 'motorcycle', 'sofa': 'couch',
           'monitor': 'tv', 'television': 'tv', 'table': 'dining table', 'plant': 'potted plant',
           'aeroplane': 'airplane', 'water bottle': 'bottle', 'computer mouse': 'mouse',
           'ball': 'sports ball', 'people': 'person', 'man': 'person', 'woman': 'person',
           'trafficlight': 'traffic light', 'stopsign': 'stop sign'}

coco_ids = {name: i for i, name in enumerate(COCO_CLASSES) if name not in ('N/A', '__background__')}


def to_coco_id(name):
    key = CLASS_MAP.get(name, name).lower().replace('_', ' ').replace('-', ' ').strip()
    key = ALIASES.get(key, key)
    if key not in coco_ids and key.endswith('s'):
        key = key[:-1]  # bottles -> bottle
    return coco_ids.get(key)


label_names = sorted({name for _, objects in samples for name, _ in objects})
unmatched = [name for name in label_names if to_coco_id(name) is None]
print('Label mapping:')
for name in label_names:
    cid = to_coco_id(name)
    print(f"  {name:20} -> {COCO_CLASSES[cid] if cid is not None else 'not a COCO class'}")

eval_images = []
n_dropped = 0
for img_path, objects in samples:
    gt = {'boxes': [], 'labels': []}
    for name, box in objects:
        cid = to_coco_id(name)
        if cid is not None:
            gt['boxes'].append(box)
            gt['labels'].append(cid)
        else:
            n_dropped += 1
    eval_images.append((img_path, gt))

if unmatched:
    print(f"\nLeft out {n_dropped} objects labelled {unmatched}: "
          "the pretrained COCO models were never trained on these classes.")

n_objects = sum(len(gt['labels']) for _, gt in eval_images)
if n_objects == 0:
    raise ValueError("None of the labels match a COCO class. Check CLASS_MAP.")
eval_classes = sorted({l for _, gt in eval_images for l in gt['labels']})
print(f"\nEvaluation set: {len(eval_images)} images, {n_objects} objects, "
      f"classes: {[COCO_CLASSES[c] for c in eval_classes]}")


# ===== Block 9: Task 3 - IoU, AP, mAP, precision and recall functions =====
def calculate_iou(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    inter_area = max(0, min(xmax1, xmax2) - max(xmin1, xmin2)) * max(0, min(ymax1, ymax2) - max(ymin1, ymin2))
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area != 0 else 0


def calculate_ap(precisions, recalls):
    precisions = [0.0] + precisions + [0.0]
    recalls = [0.0] + recalls + [1.0]
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    indices = [i for i in range(1, len(recalls)) if recalls[i] != recalls[i - 1]]
    return sum((recalls[i] - recalls[i - 1]) * precisions[i] for i in indices)


def calculate_map(predictions, ground_truths, iou_threshold=0.5):
    # Group ground truth boxes and detections by class
    gts_by_class, dets_by_class = {}, {}
    for i, gt in enumerate(ground_truths):
        for box, label in zip(gt['boxes'], gt['labels']):
            gts_by_class.setdefault(label, {}).setdefault(i, []).append(box)
    for i, pred in enumerate(predictions):
        for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
            dets_by_class.setdefault(label, []).append((score, i, box))

    average_precisions = {}
    for class_id, gt_images in gts_by_class.items():
        n_gt = sum(len(boxes) for boxes in gt_images.values())
        detections = sorted(dets_by_class.get(class_id, []), key=lambda d: -d[0])
        matched = {i: set() for i in gt_images}
        tp = fp = 0
        precisions, recalls = [], []
        for score, i, box in detections:
            best_iou, best_j = 0, -1
            for j, gt_box in enumerate(gt_images.get(i, [])):
                if j not in matched[i]:
                    iou = calculate_iou(box, gt_box)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
            if best_iou >= iou_threshold:
                tp += 1
                matched[i].add(best_j)
            else:
                fp += 1
            precisions.append(tp / (tp + fp))
            recalls.append(tp / n_gt)
        average_precisions[class_id] = calculate_ap(precisions, recalls)

    return np.mean(list(average_precisions.values())), average_precisions


def precision_recall(predictions, ground_truths, score_threshold=0.5, iou_threshold=0.5):
    # Precision and recall of the detections with a score of at least score_threshold
    tp = fp = n_gt = 0
    for pred, gt in zip(predictions, ground_truths):
        n_gt += len(gt['boxes'])
        matched = set()
        dets = sorted([d for d in zip(pred['scores'], pred['boxes'], pred['labels']) if d[0] >= score_threshold],
                      key=lambda d: -d[0])
        for score, box, label in dets:
            best_iou, best_j = 0, -1
            for j, (gt_box, gt_label) in enumerate(zip(gt['boxes'], gt['labels'])):
                if gt_label == label and j not in matched:
                    iou = calculate_iou(box, gt_box)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
            if best_iou >= iou_threshold:
                tp += 1
                matched.add(best_j)
            else:
                fp += 1
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / n_gt if n_gt else 0
    return precision, recall


# ===== Block 10: Task 3 - Run all three models and calculate mAP =====
ground_truths = [gt for _, gt in eval_images]
all_predictions, results = {}, []

for name, m in detectors.items():
    print(f"Running {name} on {len(eval_images)} images...")
    predictions = []
    total_time = 0.0
    for path, _ in eval_images:
        tensor = torch.from_numpy(np.array(open_image(path))).permute(2, 0, 1).float().div(255).to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            out = m([tensor])[0]
        if device.type == 'cuda':
            torch.cuda.synchronize()
        total_time += time.time() - start

        # Keep only the classes that were labelled; the models may also find other COCO objects
        keep = [k for k, l in enumerate(out['labels'].tolist()) if l in eval_classes]
        predictions.append({'boxes': out['boxes'][keep].cpu().tolist(),
                            'scores': out['scores'][keep].cpu().tolist(),
                            'labels': out['labels'][keep].cpu().tolist()})

    map50, ap_per_class = calculate_map(predictions, ground_truths, 0.5)
    map75, _ = calculate_map(predictions, ground_truths, 0.75)
    map_coco = np.mean([calculate_map(predictions, ground_truths, t)[0] for t in np.arange(0.5, 0.96, 0.05)])
    precision, recall = precision_recall(predictions, ground_truths)

    all_predictions[name] = (predictions, ap_per_class)
    results.append((name, map50, map75, map_coco, precision, recall, total_time / len(eval_images) * 1000))

print(f"\n{'Model':32} {'mAP@0.5':>8} {'mAP@0.75':>9} {'mAP@0.5:0.95':>13} {'Precision':>10} {'Recall':>7} {'ms/image':>9}")
for name, map50, map75, map_coco, p, r, t in results:
    print(f"{name:32} {map50:8.3f} {map75:9.3f} {map_coco:13.3f} {p:10.3f} {r:7.3f} {t:9.1f}")
print("(Precision and recall use detections with a score of at least 0.5 and an IoU of at least 0.5)")

if len(eval_classes) > 1:
    print(f"\nAP@0.5 per class:")
    print(f"{'Class':16}" + ''.join(f"{name[:28]:>30}" for name in detectors))
    for c in eval_classes:
        print(f"{COCO_CLASSES[c]:16}" + ''.join(f"{all_predictions[name][1][c]:30.3f}" for name in detectors))

x = np.arange(len(results))
plt.figure(figsize=(10, 5))
plt.bar(x - 0.25, [r[1] for r in results], width=0.25, label='mAP@0.5')
plt.bar(x, [r[2] for r in results], width=0.25, label='mAP@0.75')
plt.bar(x + 0.25, [r[3] for r in results], width=0.25, label='mAP@0.5:0.95')
plt.xticks(x, [r[0] for r in results])
plt.ylim(0, 1)
plt.ylabel('mAP')
plt.title(f'mAP on {len(eval_images)} labelled images')
plt.legend()
plt.grid(axis='y')
plt.show()


# ===== Block 11: Task 3 - Ground truth vs predictions =====
colors = ['red', 'deepskyblue', 'orange', 'magenta']
with_gt = [i for i, (_, gt) in enumerate(eval_images) if gt['boxes']]
picks = random.Random(0).sample(with_gt, min(6, len(with_gt)))

plt.figure(figsize=(18, 10))
for k, idx in enumerate(picks):
    path, gt = eval_images[idx]
    ax = plt.subplot(2, 3, k + 1)
    ax.imshow(open_image(path))
    for (x1, y1, x2, y2) in gt['boxes']:
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='lime', linewidth=3))
    for name, color in zip(detectors, colors):
        pred = all_predictions[name][0][idx]
        for (x1, y1, x2, y2), s, l in zip(pred['boxes'], pred['scores'], pred['labels']):
            if s >= 0.5:
                ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color,
                                               linewidth=1.5, linestyle='--'))
                ax.text(x1, y1 - 3, f"{COCO_CLASSES[l]} {s:.2f}", color=color, fontsize=8,
                        backgroundcolor='white')
    ax.axis('off')
plt.suptitle('Ground truth (green) vs ' + ', '.join(f'{n} ({c})' for n, c in zip(detectors, colors)))
plt.tight_layout()
plt.show()
