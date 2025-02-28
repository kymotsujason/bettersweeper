import math
import cv2
import numpy as np
import mss
import pygetwindow as gw
import pyautogui
import time
import random
from functools import lru_cache
from humancursor import SystemCursor

def debug_show_image(image, window_name="Debug", wait=True):
    """
    Utility to display an image for debugging.
    """
    cv2.imshow(window_name, image)
    if wait:
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_board_cells(img, window_region, min_cells=64, debug=False):
    """
    Uses adaptive thresholding, contour detection, and non‐maximum suppression to
    detect candidate Minesweeper cells.
    
    If debug=True, it draws candidate boxes at various stages.
    
    Returns a list of rows, where each row is a sorted list of bounding boxes (x, y, w, h).
    """
    # Convert to grayscale and apply blur.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive threshold to highlight grid lines.
    thresh = cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Dilate to close gaps.
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    # Find contours (using RETR_TREE to capture nested ones).
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if debug:
        print(f"DEBUG: Total contours found: {len(contours)}")
    
    candidate_boxes = []
    max_area = (img.shape[0] * img.shape[1]) / 10  # discard very large regions (like the board border)
    
    # First pass: candidate boxes from contours.
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        area = cv2.contourArea(approx)
        if  2 < len(approx) < 7 and 100 < area < max_area:
            x, y, w, h = cv2.boundingRect(approx)
            # Check if the bounding box is roughly square.
            if 0.8 <= (w / h) <= 1.2:
                candidate_boxes.append((x, y, w, h))
    
    if debug:
        print(f"DEBUG: Candidate boxes before NMS: {len(candidate_boxes)}")
    nms_img = img.copy()
    for (x, y, w, h) in candidate_boxes:
        cv2.rectangle(nms_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    debug_show_image(nms_img, "Post NMS Boxes", wait=False)
    
    # Optionally draw these initial boxes.
    if debug:
        debug_img = img.copy()
        for (x, y, w, h) in candidate_boxes:
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        debug_show_image(debug_img, "Initial Candidate Boxes", wait=False)
    
    # Apply non-maximum suppression (NMS) to merge overlapping boxes.
    candidate_boxes = non_max_suppression_fast(candidate_boxes, overlapThresh=0.3)
    
    if debug:
        print(f"DEBUG: Candidate boxes after NMS: {len(candidate_boxes)}")
        nms_img = img.copy()
        for (x, y, w, h) in candidate_boxes:
            cv2.rectangle(nms_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        debug_show_image(nms_img, "Post NMS Boxes", wait=False)
    
    if len(candidate_boxes) < min_cells:
        if debug:
            print(f"DEBUG: Only {len(candidate_boxes)} candidate boxes found; expected at least {min_cells}.")
        return None

    # Outlier removal based on the center cluster of candidate boxes.
    centers = [(x + w/2, y + h/2) for (x, y, w, h) in candidate_boxes]
    centers_x = [c[0] for c in centers]
    centers_y = [c[1] for c in centers]
    Q1_x, Q3_x = np.percentile(centers_x, [25, 75])
    IQR_x = Q3_x - Q1_x
    Q1_y, Q3_y = np.percentile(centers_y, [25, 75])
    IQR_y = Q3_y - Q1_y

    lower_x, upper_x = Q1_x - 1.5 * IQR_x, Q3_x + 1.5 * IQR_x
    lower_y, upper_y = Q1_y - 1.5 * IQR_y, Q3_y + 1.5 * IQR_y

    filtered_boxes = []
    for (x, y, w, h) in candidate_boxes:
        cx = x + w/2
        cy = y + h/2
        if lower_x <= cx <= upper_x and lower_y <= cy <= upper_y:
            # Titlebar, menu, and right sidebar removal
            if cx <= window_region["width"] - 400 and window_region["top"] + 100 <= cy:
                filtered_boxes.append((x, y, w, h))
    
    if debug:
        print(f"DEBUG: Candidate boxes after outlier filtering: {len(filtered_boxes)}")
        filtered_img = img.copy()
        for (x, y, w, h) in filtered_boxes:
            cv2.rectangle(filtered_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        debug_show_image(filtered_img, "Filtered Candidate Boxes", wait=True)
    
    # Group boxes into rows.
    filtered_boxes = sorted(filtered_boxes, key=lambda b: (b[1], b[0]))
    rows_list = []
    heights = [h for (_, _, _, h) in filtered_boxes]
    median_h = np.median(heights)
    row_tol = median_h * 0.5

    current_row = []
    current_row_y = None
    for box in filtered_boxes:
        x, y, w, h = box
        center_y = y + h / 2
        if current_row_y is None:
            current_row_y = center_y
            current_row.append(box)
        else:
            if abs(center_y - current_row_y) < row_tol:
                current_row.append(box)
            else:
                rows_list.append(sorted(current_row, key=lambda b: b[0]))
                current_row = [box]
                current_row_y = center_y
    if current_row:
        rows_list.append(sorted(current_row, key=lambda b: b[0]))
    
    if debug:
        print(f"DEBUG: Grouped rows count: {len(rows_list)}")
        # Optionally, draw row lines or annotate row numbers.
        group_img = img.copy()
        for row_idx, row in enumerate(rows_list):
            for (x, y, w, h) in row:
                cv2.rectangle(group_img, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(group_img, str(row_idx), (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        debug_show_image(group_img, "Grouped Rows", wait=False)
    
    return rows_list

def non_max_suppression_fast(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes).astype("float")
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(y2)
    pick = []
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]
        for pos in range(0, last):
            j = idxs[pos]
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            overlap = (w * h) / areas[j]
            if overlap > overlapThresh:
                suppress.append(pos)
        idxs = np.delete(idxs, suppress)
    return boxes[pick].astype("int").tolist()

# Example usage: Capture the Minesweeper window and debug the board detection.
def test_board_detection():
    window_title = "Minesweeper"  # Adjust window title if needed.
    windows = gw.getWindowsWithTitle(window_title)
    if not windows:
        print(f"No window found with title '{window_title}'")
        return
    window = windows[0]
    region = {
        "top": window.top,
        "left": window.left,
        "width": window.width,
        "height": window.height
    }
    with mss.mss() as sct:
        sct_img = sct.grab(region)
        board_img = np.array(sct_img)[:, :, :3]
    
    rows_list = detect_board_cells(board_img, region, min_cells=64, debug=True)
    if rows_list is None:
        print("Board detection failed (not enough candidate boxes).")
    else:
        print("Detected board rows:")
        for idx, row in enumerate(rows_list):
            print(f"Row {idx}: {row}")

def classify_cell_by_color(cell_img, color_map, tol=40, offset=2, patch_size=4):
    """
    Samples a small patch from a sweet spot within the cell and inspects each pixel.
    If any pixel is similar (within 'tol') to a target color for a number (labels 
    other than "0" or "hidden"), the function returns that number immediately.
    
    If no pixel matches a nonzero number, it falls back on an average patch
    approach to decide if the cell is "0" or "hidden" or another state.
    
    Parameters:
       cell_img: The image of the cell (in BGR format).
       color_map: Dictionary mapping labels (e.g. "hidden", "0", "1", "2", ...) to BGR colors.
       tol: Euclidean distance tolerance for color matching.
       offset: Number of pixels to shift right from the cell center to seek the "sweet spot".
       patch_size: Size (in pixels) of the square patch to check.
       
    Returns:
       The label of the cell (such as "1", "2", etc.) based on pixel matching.
    """
    h, w, _ = cell_img.shape
    center_x, center_y = w // 2, h // 2
    # Define the sweet spot: a few pixels to the right of the center.
    sweet_x = center_x + offset
    sweet_y = center_y

    # Define the patch boundaries (ensuring we stay within the cell).
    half_patch = patch_size // 2
    x1 = max(0, sweet_x - half_patch)
    y1 = max(0, sweet_y - half_patch)
    x2 = min(w, sweet_x + half_patch + 1)
    y2 = min(h, sweet_y + half_patch + 1)
    patch = cell_img[y1:y2, x1:x2]

    # Iterate through every pixel in the patch.
    # For each pixel, check against all number colors (ignoring "0" and "hidden").
    for row in patch:
        for pixel in row:
            # Convert pixel to float for distance computation.
            pixel_value = pixel.astype(np.float32)
            for label, target in color_map.items():
                if label in ["0", "hidden"]:
                    continue  # Skip non-number labels.
                target_color = np.array(target, dtype=np.float32)
                distance = np.linalg.norm(pixel_value - target_color)
                if distance <= tol:
                    return label  # Immediately return matching number.

    # Fallback: if no pixel matches a number, compute the average color of the patch.
    mean_color = np.mean(patch, axis=(0, 1))
    best_label = "unknown"
    best_distance = float("inf")
    for label, target in color_map.items():
        target_color = np.array(target, dtype=np.float32)
        distance = np.linalg.norm(mean_color - target_color)
        if distance < best_distance and distance <= tol:
            best_distance = distance
            best_label = label
    return best_label

def get_board_state(window_region, color_map, tol=40):
    with mss.mss() as sct:
        sct_img = sct.grab(window_region)
        full_img = np.array(sct_img)[:, :, :3]
    cell_rows = detect_board_cells(full_img, window_region)
    if cell_rows is None:
        return None, None
    board_state = []
    for row in cell_rows:
        row_state = []
        for (x, y, w, h) in row:
            cell_img = full_img[y:y+h, x:x+w]
            label = classify_cell_by_color(cell_img, color_map, tol)
            row_state.append(label)
        board_state.append(row_state)
    return board_state, cell_rows

def solve_board_state(board_state):
    moves = []
    rows = len(board_state)
    cols = len(board_state[0]) if rows > 0 else 0
    for i in range(rows):
        for j in range(cols):
            cell = board_state[i][j]
            # Only process revealed numbers (skip "hidden" or "unknown")
            if cell in ["hidden", "unknown"]:
                continue
            try:
                number = int(cell)
            except ValueError:
                continue
            neighbors = get_neighbors(i, j, rows, cols)
            hidden = []
            flag_count = 0
            for (ni, nj) in neighbors:
                if board_state[ni][nj] in ["hidden", "unknown"]:
                    hidden.append((ni, nj))
                elif board_state[ni][nj] == "flag":
                    flag_count += 1
            if hidden and number == flag_count + len(hidden):
                for (ni, nj) in hidden:
                    move = (ni, nj, "flag")
                    if move not in moves:
                        moves.append(move)
            if hidden and number == flag_count:
                for (ni, nj) in hidden:
                    move = (ni, nj, "click")
                    if move not in moves:
                        moves.append(move)
    return moves

################################################################################
# PROBABILITY-THRESHOLD APPROACH
################################################################################

def estimate_cell_probability_avg(board_state, i, j):
    """
    Estimate the probability that a hidden/unknown cell at (i, j) is a mine.
    Instead of taking the maximum risk from all numbered neighbors, compute the
    average risk.
    
    For each numbered neighbor:
        risk = (number - flagged_neighbors) / (total hidden neighbors)
    Returns the average risk, or a default value (e.g., 0.05) if no numbered neighbor is present.
    """
    rows = len(board_state)
    cols = len(board_state[0])
    risks = []
    for (ni, nj) in get_neighbors(i, j, rows, cols):
        try:
            number = int(board_state[ni][nj])
        except ValueError:
            continue  # Skip non-number cells.
        
        neighbor_cells = get_neighbors(ni, nj, rows, cols)
        flagged = sum(1 for (xi, xj) in neighbor_cells if board_state[xi][xj] == "flag")
        hidden_list = [ (xi, xj) for (xi, xj) in neighbor_cells if board_state[xi][xj] in ["hidden", "unknown"] ]
        if hidden_list:
            risk = max(0, number - flagged) / len(hidden_list)
            risks.append(risk)
    
    if risks:
        avg_risk = sum(risks) / len(risks)
        return avg_risk
    else:
        return 0.05  # Default low risk if no numbered neighbor.

def choose_least_risky_hidden_cell(board_state, threshold=0.1, debug=False):
    """
    Loop over all hidden (or unknown) cells, estimate the average risk for each,
    and return the cell with the lowest risk if that risk is below the `threshold`.
    """
    rows = len(board_state)
    cols = len(board_state[0])
    best_cell = None
    best_risk = float('inf')
    for i in range(rows):
        for j in range(cols):
            if board_state[i][j] in ["hidden", "unknown"]:
                risk = estimate_cell_probability_avg(board_state, i, j)
                if debug:
                    print(f"Cell ({i},{j}) estimated risk: {risk:.3f}")
                if risk < best_risk:
                    best_risk = risk
                    best_cell = (i, j)
    if best_cell and best_risk < threshold:
        return best_cell, best_risk
    else:
        return None, None


################################################################################
# ADVANCED SOLVER (with consistency checks and propagation)
################################################################################

def is_move_consistent(board_state, i, j):
    """
    Checks if revealing cell (i, j) would be consistent with its numbered neighbors.
    For each numbered neighbor, if we exclude cell (i,j) from hidden then verify that
    (number - flagged) <= (remaining hidden count). Returns True if consistent.
    """
    rows = len(board_state)
    cols = len(board_state[0])
    for (ni, nj) in get_neighbors(i, j, rows, cols):
        try:
            number = int(board_state[ni][nj])
        except ValueError:
            continue
        flagged = 0
        hidden = 0
        for (xi, xj) in get_neighbors(ni, nj, rows, cols):
            if (xi, xj) == (i, j):
                continue  # candidate cell is assumed revealed
            if board_state[xi][xj] == "flag":
                flagged += 1
            elif board_state[xi][xj] in ["hidden", "unknown"]:
                hidden += 1
        if (number - flagged) > hidden:
            return False
    return True

def get_neighbors(i, j, rows, cols):
    """Return list of neighbors' coordinates for cell (i,j)."""
    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                neighbors.append((ni, nj))
    return neighbors

###################################################
# 1. Cache for neighbors
###################################################
neighbors_cache = {}

def get_neighbors_cached(i, j, rows, cols):
    """
    Returns a list of neighbor coordinate pairs for cell (i,j) given board dimensions.
    Uses a global cache to avoid recomputation.
    """
    key = (i, j, rows, cols)
    if key not in neighbors_cache:
        neighbors_cache[key] = [(i + di, j + dj)
                                for di in (-1, 0, 1)
                                for dj in (-1, 0, 1)
                                if not (di == 0 and dj == 0) and 0 <= i + di < rows and 0 <= j + dj < cols]
    return neighbors_cache[key]

###################################################
# 2. Memoize bounding box calculations
###################################################
@lru_cache(maxsize=None)
def memoized_bounding_box(cells_tuple):
    """
    Receives a tuple of (i, j) pairs and returns a bounding box (min_i, min_j, max_i, max_j).
    The tuple must be sorted (or at least consistent) for caching to work.
    """
    min_i = min(x for x, y in cells_tuple)
    max_i = max(x for x, y in cells_tuple)
    min_j = min(y for x, y in cells_tuple)
    max_j = max(y for x, y in cells_tuple)
    return (min_i, min_j, max_i, max_j)

def compute_bounding_box(U):
    """
    Given a set U of (i, j) tuples, returns its bounding box.
    Converts U to a sorted tuple for caching.
    """
    return memoized_bounding_box(tuple(sorted(U)))

def boxes_overlap(box1, box2):
    """
    Checks if two bounding boxes (min_i, min_j, max_i, max_j) overlap.
    They do not overlap when one lies completely to the left/right or above/below the other.
    """
    # If one box's max_i is less than the other's min_i, no vertical overlap.
    if box1[2] < box2[0] or box2[2] < box1[0]:
        return False
    # Similarly for horizontal.
    if box1[3] < box2[1] or box2[3] < box1[1]:
        return False
    return True

#############################################
# Advanced Solver using Bitmask Representation
#############################################
def solve_board_state_advanced_bitmask(board_state):
    """
    Optimized Minesweeper solver that builds constraints and propagates information
    using bitmask representations.
    
    board_state is a 2D list where each cell is a string:
      - A revealed number (like "1", "2", etc.)
      - "hidden" or "unknown" for an unrevealed cell
      - "flag" if the cell has been flagged as a mine.
      
    Returns a list of moves. Each move is a tuple: (row, col, action),
    where action is "click", "flag", or "unflag".
    """
    rows = len(board_state)
    cols = len(board_state[0]) if rows > 0 else 0
    total_cells = rows * cols

    # Each cell’s linear index: index = i * cols + j.
    # Build constraints: for every revealed numbered cell, include the bitmask over unknown neighbors.
    # A constraint is represented as a tuple: (mask, remaining)
    #   - mask: an integer whose bits represent the positions of unknown neighbors.
    #   - remaining: how many mines are expected within that mask.
    constraints = []
    for i in range(rows):
        for j in range(cols):
            try:
                number = int(board_state[i][j])
            except ValueError:
                continue  # Skip non-number cells.
            nb_list = get_neighbors_cached(i, j, rows, cols)
            mask = 0
            flag_count = 0
            for (ni, nj) in nb_list:
                state = board_state[ni][nj]
                if state in ["hidden", "unknown"]:
                    idx = ni * cols + nj
                    mask |= (1 << idx)
                elif state == "flag":
                    flag_count += 1
            if mask != 0:
                remaining = number - flag_count
                if remaining < 0:
                    remaining = 0  # safeguard
                constraints.append((mask, remaining))
    
    # Sort constraints by the number of bits in the mask (i.e. number of unknown neighbors).
    constraints.sort(key=lambda c: (c[0].bit_count() if hasattr(c[0], "bit_count") else bin(c[0]).count("1")))
    
    # Basic inference: for each constraint:
    # If remaining == 0, then all cells in mask are safe.
    # If remaining equals the number of bits in mask, then all cells in mask are mines.
    safe_mask = 0
    mine_mask = 0
    for mask, rem in constraints:
        cnt = mask.bit_count() if hasattr(mask, "bit_count") else bin(mask).count("1")
        if rem == 0:
            safe_mask |= mask
        elif rem == cnt:
            mine_mask |= mask

    # Propagation using subset relations.
    changed = True
    while changed:
        changed = False
        n = len(constraints)
        # Compare each pair only once.
        for i in range(n):
            mask1, rem1 = constraints[i]
            for j in range(i+1, n):
                mask2, rem2 = constraints[j]
                # Check if mask1 is a subset of mask2:
                if mask1 & ~mask2 == 0:
                    diff = mask2 & ~mask1
                    diff_count = rem2 - rem1
                    if diff != 0:
                        # Use Python's bit_count (works for arbitrary large ints).
                        cnt_diff = diff.bit_count() if hasattr(diff, "bit_count") else bin(diff).count("1")
                        if diff_count == 0:
                            safe_mask |= diff
                            changed = True
                        elif diff_count == cnt_diff:
                            mine_mask |= diff
                            changed = True
                # Check if mask2 is a subset of mask1:
                if mask2 & ~mask1 == 0:
                    diff = mask1 & ~mask2
                    diff_count = rem1 - rem2
                    if diff != 0:
                        cnt_diff = diff.bit_count() if hasattr(diff, "bit_count") else bin(diff).count("1")
                        if diff_count == 0:
                            safe_mask |= diff
                            changed = True
                        elif diff_count == cnt_diff:
                            mine_mask |= diff
                            changed = True
        # Update constraints: remove cells that have been decided from each mask.
        new_constraints = []
        for mask, rem in constraints:
            # Remove cells that are already in safe_mask or mine_mask.
            new_mask = mask & ~(safe_mask | mine_mask)
            # Reduce the count by the number of cells that were mines.
            reduction = (mask & mine_mask).bit_count() if hasattr(mask & mine_mask, "bit_count") else bin(mask & mine_mask).count("1")
            new_rem = rem - reduction
            if new_mask and new_rem >= 0:
                new_constraints.append((new_mask, new_rem))
        constraints = new_constraints

    # Convert safe_mask and mine_mask to moves.
    moves = []
    for bit in range(total_cells):
        cell_mask = (1 << bit)
        i = bit // cols
        j = bit % cols
        if safe_mask & cell_mask:
            # If a cell was flagged but turns out safe, output an "unflag" before a "click".
            if board_state[i][j] == "flag":
                moves.append((i, j, "unflag"))
                moves.append((i, j, "click"))
            else:
                moves.append((i, j, "click"))
        if mine_mask & cell_mask:
            moves.append((i, j, "flag"))
    return moves
            

def euclidean_distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def order_moves_by_proximity(moves, cell_rows, window):
    """
    Given a list of moves (each move is a tuple: (r, c, action)),
    convert each to its on-screen pixel coordinate (using cell_rows and window),
    then reorder the moves using a greedy nearest-neighbor approach
    starting from the current mouse position.
    
    Returns the re-ordered move list.
    """
    # Convert moves to a list of tuples: (pixel_coord, (r, c, action))
    moves_with_coords = []
    for (r, c, action) in moves:
        try:
            box = cell_rows[r][c]  # box = (x, y, w, h)
        except IndexError:
            continue  # Skip moves whose indices are out-of-range
        x, y, w, h = box
        # Compute the center of the cell in screen coordinates.
        click_x = window.left + x + w // 2
        click_y = window.top + y + h // 2
        moves_with_coords.append(((click_x, click_y), (r, c, action)))
    
    # Start at the current mouse position.
    current_pos = pyautogui.position()
    ordered_moves = []
    
    # Greedy nearest-neighbor selection.
    while moves_with_coords:
        # Find the move whose pixel coordinate is closest to current_pos.
        best = min(moves_with_coords, key=lambda m: euclidean_distance(current_pos, m[0]))
        ordered_moves.append(best[1])
        current_pos = best[0]
        moves_with_coords.remove(best)
    
    return ordered_moves

def check_user_mouse_position(expected_pos, tolerance=30):
    """
    Checks if the current mouse position is within `tolerance` pixels of `expected_pos`.
    Returns True if within tolerance (i.e. user hasn't significantly moved the mouse),
    or False otherwise.
    """
    current_pos = pyautogui.position()
    dist = euclidean_distance(expected_pos, current_pos)
    if dist > tolerance:
        print(f"DEBUG: Mouse moved by {dist:.2f} pixels from expected {expected_pos}.")
        return False
    return True

################################################################################
# MAIN LOOP WITH INTEGRATED PROBABILITY THRESHOLD
################################################################################

def main():
    cursor = SystemCursor()
    window_title = "Minesweeper"  # Adjust as needed.
    windows = gw.getWindowsWithTitle(window_title)
    if not windows:
        print(f"No window found with title '{window_title}'")
        return
    window = windows[0]
    window.activate()
    time.sleep(0.2)
    
    region = {
        "top": window.top,
        "left": window.left,
        "width": window.width,
        "height": window.height
    }
    
    # Define your colour mapping for cell classification (BGR).
    color_map = {
        "hidden": [255, 213, 123],    # Hidden cell rgb(123, 213, 255)
        "flag":   [  0,   0, 198],    # Flagged rgb(198, 0, 0)
        "0":      [255, 255, 255],    # Revealed (0) rgb(255, 255, 255)
        "1":      [214, 182,  24],    # Blue (1) rgb(24, 182, 214)
        "2":      [ 20, 143, 108],    # Green (2) rgb(108, 143, 20)
        "3":      [ 74,  24, 156],    # Red (3) rgb(156, 24, 74)
        "4":      [191,  86,  30],    # Purple (4) rgb(30, 86, 191)
        "5":      [ 25,  26, 163],    # Brown (5) rgb(163, 26, 25)
        "6":      [103, 105,   0],    # Cyan (6) rgb(0, 105, 103)
    }
    
    print("Starting bot. Press Ctrl+C to stop.")
    while True:
        # Before executing the moves, record the current mouse position as the "expected" start.
        expected_pos = pyautogui.position()

        board_state, cell_rows = get_board_state(region, color_map, tol=40)
        if board_state is None or cell_rows is None:
            print("Board detection failed. Retrying...")
            time.sleep(1)
            continue
        
        # Debug: show current board state
        print("Current Board State:")
        for row in board_state:
            print(" ".join(row).replace("hidden", "h").replace("unknown", "u").replace("flag", "f"))

        start_time = time.time_ns() // 1_000_000
        # Try the advanced solver first.
        moves = solve_board_state_advanced_bitmask(board_state)
        if moves:
            print("Advanced solver moves:", moves)
        else:
            print("Advanced solver found no moves; trying basic solver.")
            moves = solve_board_state(board_state)  # Assuming you've defined a basic solver.
            if moves:
                print("Basic solver moves:", moves)
        
        # If still no moves, try the probability-threshold approach.
        if not moves:
            cell, prob = choose_least_risky_hidden_cell(board_state, threshold=0.2)
            if cell:
                print(f"Probability-threshold move: Cell {cell} with risk {prob:.2f}")
                moves.append((cell[0], cell[1], "click"))
            else:
                print("No sufficiently safe cell found by probability-threshold. Choosing random hidden cell.")
                rows_no = len(board_state)
                cols_no = len(board_state[0])
                hidden_cells = [(i, j) for i in range(rows_no) for j in range(cols_no)
                                if board_state[i][j] in ["hidden", "unknown"]]
                if hidden_cells:
                    random_cell = random.choice(hidden_cells)
                    moves.append((random_cell[0], random_cell[1], "click"))

        end_time = time.time_ns() // 1_000_000
        print(f"Processed board in {end_time - start_time:.4f} ms.")
        print("Total moves to perform:", moves)
        moves = order_moves_by_proximity(moves, cell_rows, window)
        print("Ordered moves:", moves)
        
        for move in moves:
            # Before each move, check whether the mouse is still in the expected position.
            if not check_user_mouse_position(expected_pos, tolerance=30):
                print("User moved the mouse. Aborting automated moves for safety.")
                # Optionally, you might want to pause to let the user resume, or exit entirely.
                return  # Exit the main loop (or you could break or pause)
            
            r, c, action = move
            try:
                box = cell_rows[r][c]
            except IndexError:
                continue
            x, y, w, h = box
            click_x = window.left + x + w // 2
            click_y = window.top + y + h // 2
            cursor.move_to([click_x, click_y], duration=0.05)
            if action == "unflag":
                #print(f"Unflagging cell ({r},{c}) at ({click_x},{click_y}).")
                pyautogui.doubleClick(click_x, click_y, button="right", interval=0.2)
            elif action == "click":
                #print(f"Left-clicking cell ({r},{c}) at ({click_x},{click_y}).")
                pyautogui.click(click_x, click_y, button="left")
            elif action == "flag":
                #print(f"Right-clicking cell ({r},{c}) to flag at ({click_x},{click_y}).")
                pyautogui.click(click_x, click_y, button="right")

            # Update expected_pos to the target of this move.
            expected_pos = (click_x, click_y)

        # Count the number of cells that are still hidden or unknown.
        hidden_count = sum(1 for row in board_state for cell in row if cell in ["hidden", "unknown"])

        # If the number of moves (safe clicks or flags) is equal to or greater than the number of hidden cells,
        # then we assume that every remaining cell has a deduced move.
        if moves and len(moves) >= hidden_count:
            print("Detected that the number of solver moves equals or exceeds the number of hidden cells.")
            print(f"Hidden cells: {hidden_count}, Moves: {len(moves)}. Assuming board solved; stopping bot.")
            break  # or exit the loop, or set a flag to pause further actions.
        time.sleep(0.3)

if __name__ == '__main__':
    try:
        main()
        #test_board_detection()
    except KeyboardInterrupt:
        print("Bot stopped.")