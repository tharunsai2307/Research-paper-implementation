"""
Severity Proxy and Overlapping Bounding Box Geometrical Engine.
Implements:
- Bounding Box Intersection over Union (IoU)
- Exact 2D Box Union Area calculation (Klee's Measure sweep-line algorithm) to eliminate double-counting
- Relative Affected Area Proxy calculation
"""

from typing import List, Dict, Tuple, Any

def box_area(box: List[float]) -> float:
    """Calculates area of an axis-aligned [x1, y1, x2, y2] rectangle."""
    w = max(0.0, box[2] - box[0])
    h = max(0.0, box[3] - box[1])
    return w * h

def box_iou(b1: List[float], b2: List[float]) -> float:
    """Computes Intersection over Union (IoU) between two [x1, y1, x2, y2] bounding boxes."""
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = box_area(b1)
    boxBArea = box_area(b2)
    unionArea = boxAArea + boxBArea - interArea

    return interArea / unionArea if unionArea > 0.0 else 0.0

def calculate_union_area(boxes: List[List[float]]) -> float:
    """
    Calculates exact analytical 2D union area of a collection of axis-aligned rectangles.
    Employs an exact 1D sweep-line algorithm across unique X-coordinates (Klee's measure in 2D),
    preventing any double-counting of overlapping lesion regions without grid approximation.
    Each box is expected as [x1, y1, x2, y2].
    """
    if not boxes:
        return 0.0

    valid_boxes = [b for b in boxes if b[2] > b[0] and b[3] > b[1]]
    if not valid_boxes:
        return 0.0

    # Collect unique X coordinates
    x_coords = set()
    for b in valid_boxes:
        x_coords.add(b[0])
        x_coords.add(b[2])
    sorted_x = sorted(list(x_coords))

    total_union_area = 0.0

    # Sweep across each vertical slice between adjacent X coordinates
    for i in range(len(sorted_x) - 1):
        x_left = sorted_x[i]
        x_right = sorted_x[i + 1]
        width = x_right - x_left
        if width <= 0:
            continue

        # Find active Y intervals spanning this X interval
        y_intervals = []
        for b in valid_boxes:
            if b[0] <= x_left and b[2] >= x_right:
                y_intervals.append((b[1], b[3]))

        if not y_intervals:
            continue

        # Merge overlapping 1D Y-intervals
        y_intervals.sort(key=lambda item: item[0])
        merged_y = []
        curr_start, curr_end = y_intervals[0]

        for next_start, next_end in y_intervals[1:]:
            if next_start <= curr_end:
                curr_end = max(curr_end, next_end)
            else:
                merged_y.append((curr_start, curr_end))
                curr_start, curr_end = next_start, next_end
        merged_y.append((curr_start, curr_end))

        # Sum covered Y length
        covered_y = sum(end - start for start, end in merged_y)
        total_union_area += width * covered_y

    return total_union_area

def calculate_relative_affected_area_proxy(
    boxes: List[List[float]],
    image_width: int,
    image_height: int
) -> float:
    """
    Calculates the Relative Affected Area Proxy:
        Relative Affected Area Proxy = Union(Detected Lesion Boxes) / Total Image Area

    Strictly bounded in [0.0, 1.0].
    
    IMPORTANT SCIENTIFIC NOTE:
    This is an image-space geometrical proxy reflecting visible 2D lesion footprint in the camera frame.
    It does NOT represent a validated clinical or biological agronomic disease severity score.
    """
    if not boxes or image_width <= 0 or image_height <= 0:
        return 0.0

    image_area = float(image_width * image_height)
    union_area = calculate_union_area(boxes)
    proxy = union_area / image_area

    # Clamp to [0.0, 1.0] for physical validity
    return max(0.0, min(1.0, round(proxy, 6)))
