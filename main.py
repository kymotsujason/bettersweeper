import math
import cv2
import os
import numpy as np
import mss
import pygetwindow as gw
import pyautogui
import time
import random
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import threading
from functools import lru_cache
from humancursor import SystemCursor
import pywinstyles, sys
import sv_ttk

class MinesweeperBot:
    def __init__(self, status_callback, running_callback):
        self.running = False
        self.running_callback = running_callback
        self.status_callback = status_callback
        self.thread = None

    def start(self):
        if not self.running:
            self.running_callback("running")
            self.running = True
            self.thread = threading.Thread(target=self.main)
            self.thread.daemon = True
            self.thread.start()
            self.status_callback("Bot started")

    def stop(self):
        if self.running:
            self.running_callback("stopped")
            self.running = False
            if self.thread:
                self.thread.join()  # Wait for the thread to finish
            self.status_callback("Bot stopped")

    def main(self):
        cursor = SystemCursor()
        window_title = "Microsoft Minesweeper"
        windows = gw.getWindowsWithTitle(window_title)
        if not windows:
            self.status_callback("No window found with title '{window_title}'")
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
        
        # Define colour mapping for cell classification (BGR).
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
        
        retryCount = 0
        while self.running:
            # Before executing the moves, record the current mouse position
            expected_pos = pyautogui.position()

            board_state, cell_rows = self.get_board_state(region, color_map, tol=40)
            if board_state is None or cell_rows is None:
                if (retryCount < 2):
                    retryCount += 1
                    self.status_callback("Board detection failed. Retrying...")
                    time.sleep(1)
                    continue
                else:
                    self.status_callback("Board detection failed too many times. Stopping...")
                    self.stop()
                    return

            start_time = time.time_ns() // 1_000_000
            # Try the solver first, if we have no moves, we'll have to guess
            moves = self.solve_board_state(board_state)

            if not moves:
                central_cell, distance = self.choose_center(board_state)
                if central_cell:
                    self.status_callback("Choosing central cell with distance")
                    moves.append((central_cell[0], central_cell[1], "click"))
                self.status_callback("Advanced solver found no moves, guessing...")
                cell, prob = self.guess(board_state, threshold=0.2)
                if cell:
                    self.status_callback("Probability-threshold move: Cell " + str(cell) + " with risk " + "{:.2f}".format(prob))
                    moves.append((cell[0], cell[1], "click"))
                else:
                    rows_no = len(board_state)
                    cols_no = len(board_state[0])
                    hidden_cells = [(i, j) for i in range(rows_no) for j in range(cols_no)
                                    if board_state[i][j] in ["hidden", "unknown"]]
                    if hidden_cells:
                        random_cell = random.choice(hidden_cells)
                        moves.append((random_cell[0], random_cell[1], "click"))

            end_time = time.time_ns() // 1_000_000
            final_time = format(end_time - start_time)
            self.status_callback("Thinking finished in " + final_time + " ms")
            moves = self.order_moves_by_proximity(moves, cell_rows, window)
            self.status_callback("Total moves to perform: " + str(moves))
            
            for move in moves:
                # Before each move, check whether the mouse is still in the expected position.
                if not self.check_user_mouse_position(expected_pos, tolerance=30):
                    self.status_callback("User moved the mouse, stopping for safety")
                    self.stop()
                    return
                
                r, c, action = move
                try:
                    box = cell_rows[r][c]
                except IndexError:
                    continue
                x, y, w, h = box
                click_x = window.left + x + w // 2
                click_y = window.top + y + h // 2
                cursor.move_to([click_x, click_y], duration=0.01)
                if action == "unflag":
                    pyautogui.doubleClick(click_x, click_y, button="right", interval=0.4)
                elif action == "click":
                    pyautogui.click(click_x, click_y, button="left")
                elif action == "flag":
                    pyautogui.click(click_x, click_y, button="right")

                # Update expected_pos to the target of this move.
                expected_pos = (click_x, click_y)

            # Count the number of cells that are still hidden or unknown to see if we're done
            hidden_count = sum(1 for row in board_state for cell in row if cell in ["hidden", "unknown"])
            if moves and len(moves) >= hidden_count:
                self.status_callback("Game complete, stopping bot")
                self.stop()
                return
            time.sleep(0.3)

    def choose_frontier(self, board_state):
        """
        Returns a tuple (i, j) for the hidden cell that is closest to the centroid of
        all revealed cells on the board_state.
        
        If no revealed cell exists (which shouldn't be the case after the first move),
        the function falls back to picking the cell closest to the center of the board.
        """
        rows = len(board_state)
        cols = len(board_state[0])
        
        # Collect coordinates of all revealed cells.
        revealed_cells = [
            (i, j)
            for i in range(rows)
            for j in range(cols)
            if board_state[i][j] not in ["hidden", "unknown"]
        ]
        
        # If there are no revealed cells (edge case), fall back to a central cell chooser.
        if not revealed_cells:
            center_i, center_j = rows / 2, cols / 2
            best_cell = None
            best_d = float('inf')
            for i in range(rows):
                for j in range(cols):
                    if board_state[i][j] in ["hidden", "unknown"]:
                        d = math.hypot(i - center_i, j - center_j)
                        if d < best_d:
                            best_d = d
                            best_cell = (i, j)
            return best_cell

        # Compute the centroid (average row and column) of the revealed cells.
        avg_i = sum(i for i, j in revealed_cells) / len(revealed_cells)
        avg_j = sum(j for i, j in revealed_cells) / len(revealed_cells)
        
        # Now, for each hidden cell, compute its distance to the revealed centroid.
        best_cell = None
        best_distance = float('inf')
        for i in range(rows):
            for j in range(cols):
                if board_state[i][j] in ["hidden", "unknown"]:
                    d = math.hypot(i - avg_i, j - avg_j)
                    if d < best_distance:
                        best_distance = d
                        best_cell = (i, j)
                        
        return best_cell

    def choose_center(self, board_state):
        """
        From the board_state, finds the hidden (or unknown) cell 
        that is closest to the geometric center of the board.
        
        Returns a tuple (i, j, distance) where (i, j) is the chosen cell.
        """
        rows = len(board_state)
        cols = len(board_state[0])
        center_i, center_j = rows / 2, cols / 2
        
        best_cell = None
        best_distance = float('inf')
        
        for i in range(rows):
            for j in range(cols):
                if board_state[i][j] in ["hidden", "unknown"]:
                    # Compute Euclidean distance from the cell center (approximated by its indices) to board center.
                    d = math.hypot(i - center_i, j - center_j)
                    if d < best_distance:
                        best_distance = d
                        best_cell = (i, j)
                        
        return best_cell, best_distance


    def detect_board_cells(self, img, window_region, min_cells=64):
        """
        Uses adaptive thresholding, contour detection, and non‐maximum suppression to
        detect candidate Minesweeper cells.
        
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
        
        candidate_boxes = []
        max_area = (img.shape[0] * img.shape[1]) / 10  # discard very large regions (like the board border)
        
        # Get candidate boxes from contours.
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            area = cv2.contourArea(approx)
            if  2 < len(approx) < 7 and 100 < area < max_area:
                x, y, w, h = cv2.boundingRect(approx)
                # Check if the bounding box is roughly square.
                if 0.8 <= (w / h) <= 1.2:
                    candidate_boxes.append((x, y, w, h))

        nms_img = img.copy()
        for (x, y, w, h) in candidate_boxes:
            cv2.rectangle(nms_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Apply non-maximum suppression (NMS) to merge overlapping boxes.
        candidate_boxes = self.non_max_suppression_fast(candidate_boxes, overlapThresh=0.3)

        
        if len(candidate_boxes) < min_cells:
            self.status_callback("Not enough cells found, cell detection failed")
            return None

        # Outlier removal based on the center cluster of candidate boxes
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
                # Titlebar, navbar, and sidebar removal
                if cx <= window_region["width"] - 400 and window_region["top"] + 100 <= cy:
                    filtered_boxes.append((x, y, w, h))
        
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
        return rows_list

    # https://www.geeksforgeeks.org/what-is-non-maximum-suppression/
    def non_max_suppression_fast(self, boxes, overlapThresh=0.3):
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

    def classify_cell_by_color(self, cell_img, color_map, tol=40, offset=2, patch_size=4):
        """
        Samples a small patch from a sweet spot within the cell and inspects each pixel.
        Priotises any pixel is similar (within tolerance) to a target color for a number
        
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
        # Define the sweet spot with some offset since some numbers don't have a center color
        sweet_x = center_x + offset
        sweet_y = center_y

        # Define the patch boundaries
        half_patch = patch_size // 2
        x1 = max(0, sweet_x - half_patch)
        y1 = max(0, sweet_y - half_patch)
        x2 = min(w, sweet_x + half_patch + 1)
        y2 = min(h, sweet_y + half_patch + 1)
        patch = cell_img[y1:y2, x1:x2]

        # Iterate through every pixel in the patch.
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

    def get_board_state(self, window_region, color_map, tol=40):
        with mss.mss() as sct:
            sct_img = sct.grab(window_region)
            full_img = np.array(sct_img)[:, :, :3]
        cell_rows = self.detect_board_cells(full_img, window_region)
        if cell_rows is None:
            return None, None
        board_state = []
        for row in cell_rows:
            row_state = []
            for (x, y, w, h) in row:
                cell_img = full_img[y:y+h, x:x+w]
                label = self.classify_cell_by_color(cell_img, color_map, tol)
                row_state.append(label)
            board_state.append(row_state)
        return board_state, cell_rows

    def get_neighbors(self, i, j, rows, cols):
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

    def estimate_cell_probability(self, board_state, i, j, default_risk=0.05):
        """
        Estimates the probability that a hidden cell at (i, j) is a mine by 
        combining evidence from adjacent numbered cells using a weighted 
        geometric mean approach.
        
        For each numbered neighbor:
        - risk   = (number − flagged_neighbors) / (total hidden neighbors)
        - safe   = 1 − risk
        - weight = 1 / (total hidden neighbors * distance)
                where distance is the Euclidean distance between (i,j) and the neighbor's cell.
        
        The overall safe probability is computed as:
            combined_safe = exp( (Σ [w * ln(safe)]) / (Σ w) )
        and the estimated risk is:
            estimated risk = 1 − combined_safe.
        
        If no numbered neighbors are found, returns default_risk.
        """
        rows = len(board_state)
        cols = len(board_state[0])
        neighbors = self.get_neighbors(i, j, rows, cols)
        weighted_log_safe_total = 0.0
        total_weight = 0.0
        used_any = False
        
        # Assume that each cell is one unit in dimension
        candidate_center = (i, j)
        
        for (ni, nj) in neighbors:
            try:
                number = int(board_state[ni][nj])
            except ValueError:
                # Skip cells that are not numbered (could be "hidden", "flag", etc.)
                continue
            
            # Look at the neighbor cell's own adjacent cells.
            neighbor_adjacent = self.get_neighbors(ni, nj, rows, cols)
            flagged = sum(1 for (xi, xj) in neighbor_adjacent if board_state[xi][xj] == "flag")
            hidden_neighbors = [(xi, xj) for (xi, xj) in neighbor_adjacent 
                                if board_state[xi][xj] in ["hidden", "unknown"]]
            total_hidden = len(hidden_neighbors)
            if total_hidden == 0:
                continue
            
            # Compute this neighbor's risk and safe probability.
            risk_i = max(0, number - flagged) / total_hidden
            safe_i = 1 - risk_i
            
            # Compute Euclidean distance from candidate cell center to neighbor cell center.
            distance = math.hypot(candidate_center[0] - ni, candidate_center[1] - nj)
            if distance == 0:
                distance = 1  # safeguard, though ideally candidate won't equal neighbor.
            
            weight_i = 1 / (total_hidden * distance)

            weighted_log_safe_total += weight_i * math.log(safe_i)
            total_weight += weight_i
            used_any = True

        if not used_any or total_weight == 0:
            return default_risk

        combined_safe = math.exp(weighted_log_safe_total / total_weight)
        estimated_risk = 1 - combined_safe
        return estimated_risk

    def guess(self, board_state, threshold=0.1):
        """
        Loop over all hidden (or unknown) cells, estimate the average risk for each,
        and return the cell with the lowest risk if that risk is below the `threshold`.
        If, the risk estimates are nearly equal (or no cell stands out), fall back
        to choosing a cell that is closest to the frontier (the centroid of revealed cells).
        """
        rows = len(board_state)
        cols = len(board_state[0])
        
        # Gather risk estimates for all candidate hidden/unknown cells.
        cell_risks = []
        for i in range(rows):
            for j in range(cols):
                if board_state[i][j] in ["hidden", "unknown"]:
                    risk = self.estimate_cell_probability(board_state, i, j)
                    cell_risks.append(((i, j), risk))
        
        if not cell_risks:
            return None  # No candidate available.
        
        # Find the min risk.
        cell_risks.sort(key=lambda x: x[1])
        best_cell, best_risk = cell_risks[0]
        
        # Compute the range of risks, for instance the variance or (max - min).
        risks = [r for (_, r) in cell_risks]
        risk_range = max(risks) - min(risks)
        
        # If the range is very small (we can check against a small epsilon), it means all risks are similar.
        if risk_range < 0.01:
            best_cell = self.choose_frontier(board_state)
        
        return best_cell, best_risk

    def is_move_consistent(self, board_state, i, j):
        """
        Checks if revealing cell (i, j) would be consistent with its numbered neighbors.
        For each numbered neighbor, if we exclude cell (i,j) from hidden then verify that
        (number - flagged) <= (remaining hidden count)
        """
        rows = len(board_state)
        cols = len(board_state[0])
        for (ni, nj) in self.get_neighbors(i, j, rows, cols):
            try:
                number = int(board_state[ni][nj])
            except ValueError:
                continue
            flagged = 0
            hidden = 0
            for (xi, xj) in self.get_neighbors(ni, nj, rows, cols):
                if (xi, xj) == (i, j):
                    continue  # candidate cell is revealed
                if board_state[xi][xj] == "flag":
                    flagged += 1
                elif board_state[xi][xj] in ["hidden", "unknown"]:
                    hidden += 1
            if (number - flagged) > hidden:
                return False
        return True

    neighbors_cache = {}
    def get_neighbors_cached(self, i, j, rows, cols):
        """
        Returns a list of (ni, nj) for the eight neighbors of cell (i,j)
        for a board with dimensions (rows x cols). Uses caching.
        """
        key = (i, j, rows, cols)
        if key not in self.neighbors_cache:
            self.neighbors_cache[key] = [
                (i + di, j + dj)
                for di in (-1, 0, 1)
                for dj in (-1, 0, 1)
                if not (di == 0 and dj == 0) and 0 <= i + di < rows and 0 <= j + dj < cols
            ]
        return self.neighbors_cache[key]

    @lru_cache(maxsize=None)
    def memoized_bounding_box(self, cells_tuple):
        """
        Given a sorted tuple of (i, j) cell coordinates, return the bounding box 
        as (min_i, min_j, max_i, max_j).
        """
        min_i = min(x for x, y in cells_tuple)
        max_i = max(x for x, y in cells_tuple)
        min_j = min(y for x, y in cells_tuple)
        max_j = max(y for x, y in cells_tuple)
        return (min_i, min_j, max_i, max_j)

    def compute_bounding_box(self, U):
        """
        Given a set U of (i, j) pairs, returns its bounding box.
        U is converted to a sorted tuple to enable caching.
        """
        return self.memoized_bounding_box(tuple(sorted(U)))

    def bbox_center(self, bbox):
        """
        Given a bounding box (min_i, min_j, max_i, max_j), return its center as (ci, cj)
        (using float arithmetic).
        """
        min_i, min_j, max_i, max_j = bbox
        return ((min_i + max_i) / 2.0, (min_j + max_j) / 2.0)

    def centers_distance(self, bbox1, bbox2):
        """
        Returns the Euclidean distance between the centers of two bounding boxes.
        """
        c1 = self.bbox_center(bbox1)
        c2 = self.bbox_center(bbox2)
        return math.hypot(c1[0]-c2[0], c1[1]-c2[1])

    def boxes_overlap(self, box1, box2):
        """
        Returns True if two bounding boxes (min_i, min_j, max_i, max_j) overlap.
        """
        if box1[2] < box2[0] or box2[2] < box1[0]:
            return False
        if box1[3] < box2[1] or box2[3] < box1[1]:
            return False
        return True

    def solve_board_state(self, board_state, spatial_threshold=2.5):
        """
        Given a 2D board_state where each cell is a string: a revealed number, "hidden",
        or "flag"), this function builds constraints from every revealed numbered cell and
        processes them to find safe moves and mines
        
        Constraints are built using bitmask. Each constraint is a tuple (mask, remaining, bbox)
        where:
            - mask is an int bitmask, bits represent unknown neighbor cells.
            - remaining is (number shown - flagged neighbors).
            - bbox is the bounding box (min_i, min_j, max_i, max_j) of the unknown neighbors.
        
        A spatial filtering step is used: if two constraints bounding box centers are more
        than spatial_threshold apart, we skip comparing them. Should divide the board
        
        Returns a list of moves. Each move is a tuple: (row, col, action), where action is:
            "click", "unflag", "click" or "flag".
        """
        moves = []
        rows = len(board_state)
        cols = len(board_state[0]) if rows > 0 else 0
        total_cells = rows * cols

        # Build constraints from all revealed numbered cells.
        constraints = []
        for i in range(rows):
            for j in range(cols):
                try:
                    number = int(board_state[i][j])
                except ValueError:
                    continue  # skip non-numbers
                nbs = self.get_neighbors_cached(i, j, rows, cols)
                unknown_neighbors = set()
                flag_count = 0
                for (ni, nj) in nbs:
                    state = board_state[ni][nj]
                    if state in ["hidden", "unknown"]:
                        unknown_neighbors.add((ni, nj))
                    elif state == "flag":
                        flag_count += 1
                if unknown_neighbors:
                    remaining = number - flag_count
                    if remaining < 0:
                        remaining = 0
                    bbox = self.compute_bounding_box(unknown_neighbors)
                    constraints.append([unknown_neighbors, remaining, bbox])
        
        # Sort constraints by the number of unknown neighbors.
        constraints.sort(key=lambda c: len(c[0]))
        
        # Inference: if (remaining == 0) - all in U are safe; if (remaining == len(U)) - all cells are mines.
        safe_moves = set()
        mine_moves = set()
        for U, count, bbox in constraints:
            if count == 0:
                safe_moves.update(U)
            elif count == len(U):
                mine_moves.update(U)
        
        # Constraint processing with Spatial Filtering:
        changed = True
        while changed:
            changed = False
            n = len(constraints)
            for i in range(n):
                U1, count1, bbox1 = constraints[i]
                center1 = self.bbox_center(bbox1)
                for j in range(i+1, n):
                    U2, count2, bbox2 = constraints[j]
                    center2 = self.bbox_center(bbox2)
                    # Spatial filtering: skip if centers are far apart.
                    if self.centers_distance(bbox1, bbox2) > spatial_threshold:
                        continue
                    # Also skip if they share no common cell.
                    if not U1.intersection(U2):
                        continue
                    # Check subset relations.
                    if U1.issubset(U2):
                        diff = U2 - U1
                        diff_count = count2 - count1
                        if diff:
                            if diff_count == 0 and not diff.issubset(safe_moves):
                                safe_moves.update(diff)
                                changed = True
                            if diff_count == len(diff) and not diff.issubset(mine_moves):
                                mine_moves.update(diff)
                                changed = True
                    if U2.issubset(U1):
                        diff = U1 - U2
                        diff_count = count1 - count2
                        if diff:
                            if diff_count == 0 and not diff.issubset(safe_moves):
                                safe_moves.update(diff)
                                changed = True
                            if diff_count == len(diff) and not diff.issubset(mine_moves):
                                mine_moves.update(diff)
                                changed = True
            # Update constraints by removing cells already determined.
            new_constraints = []
            for (U, count, _) in constraints:
                U_new = {u for u in U if u not in safe_moves and u not in mine_moves}
                reduction = sum(1 for u in U if u in mine_moves)
                new_count = count - reduction
                if U_new and new_count >= 0:
                    new_constraints.append([U_new, new_count, self.compute_bounding_box(U_new)])
            constraints = new_constraints
        
        # Build Move List
        for (i, j) in safe_moves:
            if board_state[i][j] == "flag":
                moves.append((i, j, "unflag"))
                moves.append((i, j, "click"))
            else:
                moves.append((i, j, "click"))
        for (i, j) in mine_moves:
            moves.append((i, j, "flag"))
        
        return moves
                

    def euclidean_distance(self, a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def order_moves_by_proximity(self, moves, cell_rows, window):
        """
        Given a list of moves (each move is a tuple: (r, c, action)),
        convert each to its on-screen pixel coordinate (using cell_rows and window),
        then reorder the moves using a greedy nearest-neighbor approach
        starting from the current mouse position.
        Could be improved, but it's not that important
        
        Returns the re-ordered move list.
        """
        # Convert moves to a list of tuples: (pixel_coord, (r, c, action))
        moves_with_coords = []
        for (r, c, action) in moves:
            try:
                box = cell_rows[r][c]
            except IndexError:
                continue
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
            best = min(moves_with_coords, key=lambda m: self.euclidean_distance(current_pos, m[0]))
            ordered_moves.append(best[1])
            current_pos = best[0]
            moves_with_coords.remove(best)
        
        return ordered_moves

    def check_user_mouse_position(self, expected_pos, tolerance=30):
        """
        Checks if the current mouse position is moved.
        """
        current_pos = pyautogui.position()
        dist = self.euclidean_distance(expected_pos, current_pos)
        if dist > tolerance:
            return False
        return True

class BotGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BetterSweeper")
        self.geometry("500x400")
        self.iconbitmap(default=os.path.join(sys.prefix if hasattr(sys, "frozen") else os.path.dirname(__file__), "icon.ico"))
        self.create_widgets()
        self.bot = MinesweeperBot(self.update_status, self.update_buttons)

    def create_widgets(self):
        button_frame = ttk.Frame(self)
        button_frame.pack(pady=10)

        self.start_button = ttk.Button(button_frame, text="Start", command=self.start_bot, width=15)
        self.start_button.grid(row=0, column=0, padx=5)

        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_bot, width=15)
        self.stop_button.grid(row=0, column=1, padx=5)
        self.stop_button["state"] = tk.DISABLED

        self.status_text = scrolledtext.ScrolledText(wrap=tk.WORD, height=15)
        self.status_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    def update_buttons(self, status):
        # Use after to ensure thread-safe GUI updates.
        self.after(0, self._update_buttons, status)
    
    def _update_buttons(self, status):
        if (status == "running"):
            self.stop_button["state"] = tk.NORMAL
            self.start_button["state"] = tk.DISABLED
        else:
            self.stop_button["state"] = tk.DISABLED
            self.start_button["state"] = tk.NORMAL
        

    def update_status(self, message):
        # Use after to ensure thread-safe GUI updates.
        self.after(0, self._update_status, message)

    def _update_status(self, message):
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)

    def start_bot(self):
        self.update_status("Starting bot...")
        self.bot.start()

    def stop_bot(self):
        self.update_status("Stopping bot...")
        self.bot.stop()

def apply_theme_to_titlebar(root):
    pywinstyles.change_header_color(root, "#1c1c1c" if sv_ttk.get_theme() == "dark" else "#fafafa")

if __name__ == '__main__':
    app = BotGUI()
    sv_ttk.set_theme("dark")
    apply_theme_to_titlebar(app)
    app.mainloop()