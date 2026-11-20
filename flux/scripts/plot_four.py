#!/usr/bin/env python3
"""Draw verified Figure 2 data in the submitted artwork's original layout."""

from base64 import b64decode, b64encode
from copy import deepcopy
from io import BytesIO
import re
import xml.etree.ElementTree as ET

from PIL import Image

SVG = "http://www.w3.org/2000/svg"
HREF = "{http://www.w3.org/1999/xlink}href"
ET.register_namespace("", SVG)
ET.register_namespace("xlink", HREF[1:-5])
ET.register_namespace("inkscape", "http://www.inkscape.org/namespaces/inkscape")
ET.register_namespace("sodipodi", "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd")

# SVG path IDs in the submitted template. Each directed edge uses a shaft
# and a head, apart from the original missing shaft for (2, 3) in panel b.
COUPLING_PATHS = dict(zip(
    ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)), range(1, 7),
))
BIPARTITE_PATHS = dict(zip(
    ((0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 0),
     (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)),
    [(n, n + 1) for n in range(66, 86, 2)] + [(0, 86)]
    + [(n, n + 1) for n in range(87, 95, 2)],
))
COLLAPSED_PATHS = dict(zip(
    ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 0), (2, 1), (2, 3),
     (3, 0), (3, 1), (3, 2)),
    [(n, n + 1) for n in range(29, 51, 2)],
))


def palette_image(element):
    encoded = element.get(HREF).split(",", 1)[1]
    with Image.open(BytesIO(b64decode(encoded))) as image:
        return image.convert("RGB")


def color_at(image, fraction):
    if not 0 <= fraction <= 1:
        raise ValueError("Figure data exceed the original color-bar limits")
    rgb = image.getpixel((image.width // 2, round((1 - fraction) * (image.height - 1))))
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def visible_mi_palette(element):
    # Keep positive near-threshold edges legible against white. The same
    # light-end adjustment applies to the paths and the displayed scale.
    palette = palette_image(element).point([
        round(level - reduction * (level / 255)**8)
        for reduction in (28, 18, 4) for level in range(256)
    ])
    stream = BytesIO()
    palette.save(stream, format="PNG")
    element.set(HREF, "data:image/png;base64," + b64encode(stream.getvalue()).decode())
    return palette


def color_path(element, color):
    style = re.sub(r"((?:fill|stroke):)#[0-9a-fA-F]{6}", rf"\g<1>{color}", element.get("style"))
    element.set("style", style)


def leading_coordinates(element):
    return tuple(float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", element.get("d"))[:2])


def scale(element):
    return tuple(float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", element.get("transform")))


def update_artwork(source, result):
    tree = ET.parse(source)
    root = tree.getroot()
    elements = {e.get("id"): e for e in root.iter() if e.get("id")}
    layer = elements["g1"]
    coupling = result["coupling_hij"]
    mi = result["mi_bits"]
    mij = result["mij_bits"]
    pairs = {(i, j) for i in range(4) for j in range(i + 1, 4) if coupling[i][j] != 0}
    bipartite = set(map(tuple, result["bipartite_displayed_edges"]))
    collapsed = set(map(tuple, result["displayed_edges"]))
    if pairs != COUPLING_PATHS.keys() or bipartite != BIPARTITE_PATHS.keys() or collapsed != COLLAPSED_PATHS.keys():
        raise ValueError("The submitted artwork has a different set of edges")

    h_palette = palette_image(elements["image12"])
    b_palette = visible_mi_palette(elements["image114"])
    d_palette = visible_mi_palette(elements["image56"])
    maximum_mi = max(max(row) for row in mi)
    for (i, j), path_id in COUPLING_PATHS.items():
        color_path(elements[f"path{path_id}"], color_at(h_palette, coupling[i][j] / 0.02))
    for (i, j), (shaft_id, head_id) in BIPARTITE_PATHS.items():
        head = elements[f"path{head_id}"]
        if shaft_id:
            shaft = elements[f"path{shaft_id}"]
        else:
            # This edge climbs one row, like path76, one row higher.
            shaft = deepcopy(elements["path76"])
            shaft.set("id", "phi2-f3-shaft")
            shaft.set("d", "m 302.583,73.266 c 49.518,-15.942 98.54,-31.727 147.066,-47.348")
            layer.insert(list(layer).index(head), shaft)
        color = color_at(b_palette, mi[i][j] / maximum_mi)
        color_path(shaft, color)
        color_path(head, color)
    for (i, j), (shaft_id, head_id) in COLLAPSED_PATHS.items():
        color = color_at(d_palette, mi[i][j] / 0.64)
        color_path(elements[f"path{shaft_id}"], color)
        color_path(elements[f"path{head_id}"], color)

    # Keep the original node label positions with renderers that do not
    # support multiple per-character shifts in one tspan.
    for label_id, separation in (("text11", 157.77), ("text55", 146.33)):
        label = elements[label_id]
        for span in list(label):
            right = deepcopy(span)
            del span.attrib["dx"]
            del right.attrib["dx"]
            right.set("id", f"{span.get('id')}-right")
            right.set("x", str(separation))
            span.text, right.text = span.text
            label.insert(list(label).index(span) + 1, right)

    # The grid already places ticks at 2.5, 5, ..., 20, in units of 10^-3.
    # Use those grid lines to map the six values to their plotted coordinates.
    left_grid, right_grid = elements["path146"], elements["path160"]
    x_scale = (leading_coordinates(right_grid)[0] - leading_coordinates(left_grid)[0]) / 17.5
    x_origin = leading_coordinates(right_grid)[0] - 20 * x_scale
    y_zero, y_top = elements["path164"], elements["path172"]
    y_origin = -leading_coordinates(y_zero)[1]
    y_scale = (-leading_coordinates(y_top)[1] - y_origin) / 0.08
    svg_x_scale, svg_y_scale = scale(left_grid)
    markers = [e for e in layer if e.get("id", "").startswith("g134")]
    if len(markers) != len(pairs):
        raise ValueError("Expected six scatter markers in the submitted artwork")
    for marker, (i, j) in zip(markers, sorted(pairs, key=lambda pair: coupling[pair[0]][pair[1]])):
        x = (x_origin + 1000 * coupling[i][j] * x_scale) * svg_x_scale
        y = (y_origin + mij[i][j] * y_scale) * svg_y_scale
        point = ET.Element(f"{{{SVG}}}circle", {
            "id": f"pair-{i}-{j}", "cx": f"{x:.7f}", "cy": f"{y:.7f}",
            "r": "3.35", "fill": "#000080",
        })
        layer.insert(list(layer).index(marker), point)
        layer.remove(marker)

    for path_id, label_id, tick in [(146, 147, "2.5"), (150, 151, "7.5"),
                                    (154, 155, "12.5"), (158, 159, "17.5")]:
        label = elements[f"text{label_id}"]
        span = label[0]
        x = leading_coordinates(elements[f"path{path_id}"])[0] * svg_x_scale
        a, b, c, d, _, y = scale(label)
        label.set("transform", f"matrix({a},{b},{c},{d},{x:.7f},{y})")
        span.set("style", span.get("style").replace("#000000", "#0000ff") + ";text-anchor:middle")
        span.text = tick
    return tree
