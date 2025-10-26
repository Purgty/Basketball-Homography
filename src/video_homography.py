"""
Basketball Court Homography Mapping with Advanced Smoothing
Processes video to map player positions onto court template with minimal jitter
"""

import cv2
import numpy as np
from collections import deque, defaultdict
from ultralytics import YOLO
from typing import Tuple, Optional, List, Dict
import os
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model paths
COURT_MODEL_PATH = r'C:\Users\aswin\OneDrive\Desktop\Sandbox\bb_homography\Models\court_model\weights\best.pt'
PLAYER_MODEL_PATH = r'C:\Users\aswin\OneDrive\Desktop\Sandbox\bb_homography\Models\player_model\weights\best.pt'
COURT_MAP_PATH = r'C:\Users\aswin\OneDrive\Desktop\Sandbox\bb_homography\court_data_inference\full-court.jpeg'

# Video paths
INPUT_VIDEO = r'C:\Users\aswin\OneDrive\Desktop\Sandbox\bb_homography\court_data_inference\clip.webm'
OUTPUT_VIDEO = r'C:\Users\aswin\OneDrive\Desktop\Sandbox\bb_homography\Output_Videos\claude_output.mp4'

# Court dimensions
COURT_WIDTH = 612
COURT_HEIGHT = 364

# Keypoint indices
LEFT_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
RIGHT_INDICES = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
CENTER_INDICES = [15, 16, 17]

# Expected positions for side detection (based on typical camera angle)
# Left side: keypoints should be on left side of frame
# Right side: keypoints should be on right side of frame
EXPECTED_LEFT_REGION = (0, 0.45)  # x-range normalized [0-1]
EXPECTED_RIGHT_REGION = (0.55, 1.0)  # x-range normalized [0-1]

# Destination points for homography
DST_POINTS = {
    "left": np.array([
        [29.0, 32.0], [30.0, 53.0], [30.0, 133.0], [29.0, 230.0], [29.0, 312.0],
        [28.0, 329.0], [62.0, 181.0], [86.0, 52.0], [91.0, 311.0], [143.0, 134.0],
        [143.0, 181.0], [142.0, 230.0], [190.0, 34.0], [194.0, 177.0], [193.0, 330.0]
    ], dtype=np.float32),
    "right": np.array([
        [419.0, 34.0], [418.0, 183.0], [418.0, 330.0], [468.0, 135.0], [468.0, 184.0],
        [470.0, 228.0], [521.0, 55.0], [522.0, 311.0], [550.0, 181.0], [584.0, 32.0],
        [585.0, 54.0], [581.0, 134.0], [582.0, 229.0], [583.0, 311.0], [583.0, 330.0]
    ], dtype=np.float32),
    "center": np.array([
        [305.0, 35.0], [306.0, 182.0], [305.0, 330.0]
    ], dtype=np.float32)
}

# Thresholds
KEYPOINT_CONF_THRESHOLD = 0.5
PLAYER_CONF_THRESHOLD = 0.5
MIN_KEYPOINTS_FOR_HOMOGRAPHY = 4

# Smoothing parameters - INCREASED for smoother output
HOMOGRAPHY_BUFFER_SIZE = 5  # Increased from 5
POSITION_BUFFER_SIZE = 5     # Increased from 3
SIDE_BUFFER_SIZE = 5        # Increased from 7
EMA_ALPHA = 0.3              # Exponential moving average weight for positions

# Player tracking
MAX_PLAYER_DISAPPEAR_FRAMES = 2  # Retain position for 2 frames if not detected
PLAYER_MATCH_THRESHOLD = 100.0   # Max distance to match players across frames


# ============================================================================
# SMOOTHING UTILITIES
# ============================================================================

class OneEuroFilter:
    """
    One Euro Filter for low-latency smoothing with reduced jitter
    Better than simple moving average for real-time applications
    """
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = 0
        
    def __call__(self, x, t):
        """Apply filter to value x at time t"""
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
            
        # Time difference
        dt = t - self.t_prev
        if dt <= 0:
            dt = 1
            
        # Estimate derivative
        dx = (x - self.x_prev) / dt
        
        # Smooth derivative
        alpha_d = self._alpha(dt, self.d_cutoff)
        dx_smooth = alpha_d * dx + (1 - alpha_d) * self.dx_prev
        
        # Smooth value
        cutoff = self.min_cutoff + self.beta * abs(dx_smooth)
        alpha = self._alpha(dt, cutoff)
        x_smooth = alpha * x + (1 - alpha) * self.x_prev
        
        # Update state
        self.x_prev = x_smooth
        self.dx_prev = dx_smooth
        self.t_prev = t
        
        return x_smooth
    
    def _alpha(self, dt, cutoff):
        """Calculate alpha based on cutoff frequency"""
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)


class HomographySmoothingAdvanced:
    """Advanced homography smoothing with outlier rejection"""
    
    def __init__(self, buffer_size: int = 15, outlier_threshold: float = 0.3):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.outlier_threshold = outlier_threshold
        self.median_H = None
        
    def add(self, H: np.ndarray) -> np.ndarray:
        """Add homography matrix with outlier detection"""
        if H is None:
            return self.get_current()
        
        # Check if H is an outlier compared to recent matrices
        if len(self.buffer) >= 3:
            recent_matrices = list(self.buffer)[-5:]
            if self._is_outlier(H, recent_matrices):
                print("Outlier homography detected, using previous")
                return self.get_current()
        
        self.buffer.append(H.copy())
        return self.get_current()
    
    def _is_outlier(self, H: np.ndarray, reference_matrices: List[np.ndarray]) -> bool:
        """Check if homography is significantly different from recent ones"""
        if not reference_matrices:
            return False
        
        # Calculate Frobenius norm differences
        diffs = []
        for ref_H in reference_matrices:
            diff = np.linalg.norm(H - ref_H, 'fro')
            diffs.append(diff)
        
        mean_diff = np.mean(diffs)
        # If difference is too large, it's an outlier
        return mean_diff > self.outlier_threshold
    
    def get_current(self) -> Optional[np.ndarray]:
        """Get smoothed homography using weighted average"""
        if not self.buffer:
            return None
        
        # Use exponential weights - recent frames have more influence
        weights = np.exp(np.linspace(-2, 0, len(self.buffer)))
        weights = weights / np.sum(weights)
        
        # Weighted average
        matrices = np.array(list(self.buffer))
        avg_H = np.average(matrices, axis=0, weights=weights)
        
        # Normalize
        avg_H = avg_H / avg_H[2, 2]
        return avg_H
    
    def reset(self):
        """Clear buffer"""
        self.buffer.clear()


class PlayerTracker:
    """
    Track players across frames with temporal smoothing
    Retains positions when players disappear briefly
    """
    
    def __init__(self, max_disappear_frames: int = 2, match_threshold: float = 100.0):
        self.max_disappear_frames = max_disappear_frames
        self.match_threshold = match_threshold
        self.players = {}  # player_id -> player_info
        self.next_player_id = 0
        self.filters = {}  # player_id -> (x_filter, y_filter)
        
    def update(self, current_positions: np.ndarray, frame_idx: int) -> Dict[int, np.ndarray]:
        """
        Update player tracking with current detections
        
        Args:
            current_positions: (N, 1, 2) array of detected positions
            frame_idx: Current frame number
            
        Returns:
            Dictionary of player_id -> smoothed_position
        """
        # Convert to (N, 2) for easier handling
        if current_positions.size == 0:
            current_positions = np.array([]).reshape(0, 2)
        else:
            current_positions = current_positions.reshape(-1, 2)
        
        # Match current detections to existing players
        matched_players = set()
        matched_detections = set()
        
        if len(self.players) > 0 and len(current_positions) > 0:
            # Calculate distances between existing players and new detections
            existing_positions = np.array([
                p['position'] for p in self.players.values()
            ])
            
            distances = cdist(existing_positions, current_positions)
            
            # Greedy matching: assign closest pairs
            while distances.size > 0:
                min_idx = np.unravel_index(distances.argmin(), distances.shape)
                min_dist = distances[min_idx]
                
                if min_dist > self.match_threshold:
                    break
                
                player_ids = list(self.players.keys())
                player_id = player_ids[min_idx[0]]
                detection_idx = min_idx[1]
                
                # Match found
                matched_players.add(player_id)
                matched_detections.add(detection_idx)
                
                # Update player
                new_pos = current_positions[detection_idx]
                self._update_player(player_id, new_pos, frame_idx)
                
                # Remove matched pair from distance matrix
                distances = np.delete(distances, min_idx[0], axis=0)
                distances = np.delete(distances, min_idx[1], axis=1)
        
        # Handle unmatched existing players (disappeared)
        for player_id in list(self.players.keys()):
            if player_id not in matched_players:
                self.players[player_id]['frames_since_seen'] += 1
                
                # Remove if disappeared too long
                if self.players[player_id]['frames_since_seen'] > self.max_disappear_frames:
                    del self.players[player_id]
                    if player_id in self.filters:
                        del self.filters[player_id]
        
        # Handle unmatched detections (new players)
        for i, pos in enumerate(current_positions):
            if i not in matched_detections:
                self._add_new_player(pos, frame_idx)
        
        # Return all tracked positions (including retained ones)
        result = {}
        for player_id, info in self.players.items():
            result[player_id] = info['smoothed_position']
        
        return result
    
    def _update_player(self, player_id: int, position: np.ndarray, frame_idx: int):
        """Update existing player with new position"""
        # Initialize filters if needed
        if player_id not in self.filters:
            self.filters[player_id] = (
                OneEuroFilter(min_cutoff=1.0, beta=0.007),
                OneEuroFilter(min_cutoff=1.0, beta=0.007)
            )
        
        # Apply One Euro Filter
        x_filter, y_filter = self.filters[player_id]
        smoothed_x = x_filter(position[0], frame_idx)
        smoothed_y = y_filter(position[1], frame_idx)
        smoothed_pos = np.array([smoothed_x, smoothed_y])
        
        self.players[player_id]['position'] = position.copy()
        self.players[player_id]['smoothed_position'] = smoothed_pos
        self.players[player_id]['last_seen'] = frame_idx
        self.players[player_id]['frames_since_seen'] = 0
    
    def _add_new_player(self, position: np.ndarray, frame_idx: int):
        """Add new player to tracking"""
        player_id = self.next_player_id
        self.next_player_id += 1
        
        self.players[player_id] = {
            'position': position.copy(),
            'smoothed_position': position.copy(),
            'last_seen': frame_idx,
            'frames_since_seen': 0
        }
    
    def get_positions_array(self) -> np.ndarray:
        """Get all smoothed positions as array (N, 1, 2)"""
        if not self.players:
            return np.array([]).reshape(0, 1, 2)
        
        positions = np.array([
            p['smoothed_position'] for p in self.players.values()
        ])
        return positions.reshape(-1, 1, 2).astype(np.float32)


class SideDetectionRobust:
    """
    Robust side detection using spatial distribution analysis
    Analyzes where keypoints cluster in the frame
    """
    
    def __init__(self, buffer_size: int = 15, image_width: int = 640):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.image_width = image_width
        
    def detect_side(self, kpts: np.ndarray, confs: np.ndarray, threshold: float = 0.5) -> str:
        """
        Detect court side using spatial distribution and geometric analysis
        
        Strategy:
        1. Analyze horizontal position of high-confidence keypoints
        2. Check if left-side keypoints are in left region of frame
        3. Check if right-side keypoints are in right region of frame
        4. Use weighted voting based on confidence and position match
        """
        # Normalize x-coordinates
        norm_x = kpts[:, 0] / self.image_width
        
        # Calculate scores for left side hypothesis
        left_score = 0
        left_weight = 0
        for i in LEFT_INDICES:
            if i < len(confs) and confs[i] > threshold:
                # Check if keypoint is in expected left region
                if EXPECTED_LEFT_REGION[0] <= norm_x[i] <= EXPECTED_LEFT_REGION[1]:
                    left_score += confs[i]
                left_weight += confs[i]
        
        # Calculate scores for right side hypothesis
        right_score = 0
        right_weight = 0
        for i in RIGHT_INDICES:
            if i < len(confs) and confs[i] > threshold:
                # Check if keypoint is in expected right region
                if EXPECTED_RIGHT_REGION[0] <= norm_x[i] <= EXPECTED_RIGHT_REGION[1]:
                    right_score += confs[i]
                right_weight += confs[i]
        
        # Additional check: center keypoints should be near center
        center_consistency = 0
        for i in CENTER_INDICES:
            if i < len(confs) and confs[i] > threshold:
                if 0.35 <= norm_x[i] <= 0.65:  # Near center
                    center_consistency += confs[i]
        
        # Normalize scores
        left_score = (left_score / max(left_weight, 1e-6)) if left_weight > 0 else 0
        right_score = (right_score / max(right_weight, 1e-6)) if right_weight > 0 else 0
        
        # Add center consistency bonus to both
        left_score += center_consistency * 0.2
        right_score += center_consistency * 0.2
        
        # Also check mean position of detected keypoints
        detected_indices = [i for i in range(len(confs)) if confs[i] > threshold]
        if detected_indices:
            mean_x = np.mean([norm_x[i] for i in detected_indices])
            # If mean is on left, boost left score
            if mean_x < 0.45:
                left_score *= 1.2
            elif mean_x > 0.55:
                right_score *= 1.2
        
        side = "left" if left_score >= right_score else "right"
        
        print(f"Side detection - Left score: {left_score:.3f}, Right score: {right_score:.3f} => {side.upper()}")
        
        return side
    
    def add_and_get_stable_side(self, side: str) -> str:
        """Add detected side and return stable side using majority voting"""
        self.buffer.append(side)
        
        # Count votes in buffer
        left_count = self.buffer.count("left")
        right_count = self.buffer.count("right")
        
        # Require strong majority for side changes (70%)
        total = left_count + right_count
        if total > 0:
            left_ratio = left_count / total
            if left_ratio > 0.7:
                return "left"
            elif left_ratio < 0.3:
                return "right"
        
        # If no strong majority, keep previous stable side
        if len(self.buffer) > 0:
            return self.buffer[-1]
        
        return side


# ============================================================================
# DETECTION FUNCTIONS
# ============================================================================

def get_player_feet(player_result, threshold: float = 0.5) -> np.ndarray:
    """
    Extract player feet positions from detection results
    
    Args:
        player_result: YOLO detection result
        threshold: Minimum confidence threshold
        
    Returns:
        Array of feet positions (N, 1, 2) or empty array with correct shape
    """
    feet_points = []
    
    if not hasattr(player_result, 'boxes') or player_result.boxes is None:
        return np.array([], dtype=np.float32).reshape(0, 1, 2)
    
    for box in player_result.boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = box
        if conf >= threshold:
            # Feet position at bottom center of bounding box
            cx = (x1 + x2) / 2
            cy = y2
            feet_points.append([cx, cy])
    
    if len(feet_points) == 0:
        return np.array([], dtype=np.float32).reshape(0, 1, 2)
    
    return np.array(feet_points, dtype=np.float32).reshape(-1, 1, 2)


# ============================================================================
# HOMOGRAPHY COMPUTATION
# ============================================================================

def compute_homography(
    kpts: np.ndarray,
    confs: np.ndarray,
    side: str,
    threshold: float = 0.5
) -> Tuple[Optional[np.ndarray], List[int], np.ndarray]:
    """
    Compute homography matrix from detected keypoints
    
    Args:
        kpts: Keypoint coordinates (N, 2)
        confs: Keypoint confidences (N,)
        side: Court side ("left" or "right")
        threshold: Minimum confidence threshold
        
    Returns:
        Tuple of (homography_matrix, used_indices, used_dst_points)
    """
    # Determine which keypoints to use
    side_indices = LEFT_INDICES if side == "left" else RIGHT_INDICES
    used_indices = side_indices + CENTER_INDICES
    
    # Get corresponding destination points
    try:
        used_dst = np.vstack([DST_POINTS[side], DST_POINTS["center"]])
    except KeyError as e:
        print(f"Error: Missing destination points for {side}: {e}")
        return None, [], np.array([])
    
    # Collect valid source and destination points
    src_pts = []
    dst_pts = []
    
    for i, dst_pt in zip(used_indices, used_dst):
        if i < len(confs) and i < len(kpts) and confs[i] > threshold:
            src_pts.append(kpts[i])
            dst_pts.append(dst_pt)
    
    # Check if we have enough points
    if len(src_pts) < MIN_KEYPOINTS_FOR_HOMOGRAPHY:
        print(f"Insufficient keypoints: {len(src_pts)} < {MIN_KEYPOINTS_FOR_HOMOGRAPHY}")
        return None, used_indices, used_dst
    
    # Compute homography with RANSAC for robustness
    src_array = np.array(src_pts, dtype=np.float32)
    dst_array = np.array(dst_pts, dtype=np.float32)
    
    H, mask = cv2.findHomography(
        src_array,
        dst_array,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=2.0,  # Reduced for tighter fit
        maxIters=2000,
        confidence=0.999
    )
    
    if H is None:
        print("Homography computation failed")
        return None, used_indices, used_dst
    
    # Normalize homography
    H = H / H[2, 2]
    
    print(f"Homography computed with {len(src_pts)} points")
    return H, used_indices, used_dst


# ============================================================================
# VISUALIZATION
# ============================================================================

def draw_visualization(
    frame: np.ndarray,
    court_canvas: np.ndarray,
    kpts: np.ndarray,
    confs: np.ndarray,
    feet_points: np.ndarray,
    projected_feet: np.ndarray,
    used_indices: List[int],
    used_dst: np.ndarray,
    side: str,
    num_tracked_players: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create visualization with keypoints and projected positions
    """
    vis_frame = frame.copy()
    vis_canvas = court_canvas.copy()
    
    # Draw destination keypoints on canvas (blue)
    for i, dst_pt in zip(used_indices, used_dst):
        x, y = int(dst_pt[0]), int(dst_pt[1])
        x = np.clip(x, 0, vis_canvas.shape[1] - 1)
        y = np.clip(y, 0, vis_canvas.shape[0] - 1)
        cv2.circle(vis_canvas, (x, y), 4, (255, 100, 0), -1)
        cv2.putText(
            vis_canvas, f"{i}", (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1
        )
    
    # Draw projected player positions on canvas (bright green with glow effect)
    if projected_feet is not None and projected_feet.size > 0:
        for pt in projected_feet:
            x, y = int(pt[0][0]), int(pt[0][1])
            x = np.clip(x, 0, vis_canvas.shape[1] - 1)
            y = np.clip(y, 0, vis_canvas.shape[0] - 1)
            
            # Glow effect
            cv2.circle(vis_canvas, (x, y), 18, (0, 200, 0), 2)
            cv2.circle(vis_canvas, (x, y), 12, (0, 255, 0), -1)
            cv2.circle(vis_canvas, (x, y), 12, (255, 255, 255), 2)
    
    # Draw player feet on original frame (red)
    if feet_points is not None and feet_points.size > 0:
        for pt in feet_points:
            cx, cy = int(pt[0][0]), int(pt[0][1])
            cv2.circle(vis_frame, (cx, cy), 8, (0, 0, 255), -1)
            cv2.circle(vis_frame, (cx, cy), 8, (255, 255, 255), 2)
    
    # Draw detected court keypoints on original frame (yellow)
    for i, (x, y) in enumerate(kpts):
        if i < len(confs) and confs[i] > KEYPOINT_CONF_THRESHOLD:
            x_int, y_int = int(x), int(y)
            cv2.circle(vis_frame, (x_int, y_int), 4, (0, 255, 255), -1)
            cv2.putText(
                vis_frame, f"{i}", (x_int + 5, y_int - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1
            )
    
    # Add info overlay on canvas
    cv2.rectangle(vis_canvas, (10, 10), (250, 80), (0, 0, 0), -1)
    cv2.rectangle(vis_canvas, (10, 10), (250, 80), (0, 255, 0), 2)
    
    cv2.putText(
        vis_canvas, f"Side: {side.upper()}", (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )
    cv2.putText(
        vis_canvas, f"Players: {num_tracked_players}", (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
    )
    
    # Add frame label
    cv2.putText(
        vis_frame, "Player Detection", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
    )
    
    return vis_frame, vis_canvas


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_video(
    input_path: str,
    output_path: str,
    court_model: YOLO,
    player_model: YOLO,
    court_map: np.ndarray
):
    """
    Process video with advanced homography mapping and smoothing
    """
    # Initialize video capture
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {total_frames} frames at {fps} FPS")
    
    # Initialize smoothing components
    homography_smoother = HomographySmoothingAdvanced(HOMOGRAPHY_BUFFER_SIZE)
    player_tracker = PlayerTracker(MAX_PLAYER_DISAPPEAR_FRAMES, PLAYER_MATCH_THRESHOLD)
    side_detector = SideDetectionRobust(SIDE_BUFFER_SIZE)
    
    # Prepare output video writer
    frame_height = 640
    frame_width = 640
    output_width = frame_width + COURT_WIDTH
    output_height = max(frame_height, COURT_HEIGHT)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    frame_idx = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            if frame_idx % 30 == 0:
                print(f"\nProcessing frame {frame_idx}/{total_frames} "
                      f"({100*frame_idx/total_frames:.1f}%)")
            
            # Resize frame
            frame = cv2.resize(frame, (frame_width, frame_height))
            
            # Detect court keypoints
            court_result = court_model(frame, verbose=False)[0]
            
            if (not hasattr(court_result, 'keypoints') or 
                court_result.keypoints is None or 
                court_result.keypoints.xy.shape[0] == 0):
                print("No court keypoints detected")
                canvas_resized = cv2.resize(court_map, (COURT_WIDTH, output_height))
                combined = np.hstack([frame, canvas_resized])
                out.write(combined)
                continue
            
            kpts = court_result.keypoints.xy[0].cpu().numpy()
            confs = court_result.keypoints.conf[0].cpu().numpy()
            
            # Detect court side with robust method
            detected_side = side_detector.detect_side(kpts, confs, KEYPOINT_CONF_THRESHOLD)
            stable_side = side_detector.add_and_get_stable_side(detected_side)
            
            # Compute homography
            H, used_indices, used_dst = compute_homography(
                kpts, confs, stable_side, KEYPOINT_CONF_THRESHOLD
            )
            
            # Apply homography smoothing
            if H is not None:
                H_smooth = homography_smoother.add(H)
            else:
                H_smooth = homography_smoother.get_current()
            
            if H_smooth is None:
                print("No valid homography available")
                canvas_resized = cv2.resize(court_map, (COURT_WIDTH, output_height))
                combined = np.hstack([frame, canvas_resized])
                out.write(combined)
                continue
            
            # Detect players
            player_result = player_model(frame, verbose=False)[0]
            feet_points = get_player_feet(player_result, PLAYER_CONF_THRESHOLD)
            
            # Update player tracking
            tracked_positions = player_tracker.update(feet_points, frame_idx)
            
            # Get smoothed positions as array
            smoothed_feet = player_tracker.get_positions_array()
            
            # Project using smoothed homography
            if smoothed_feet.size == 0:
                projected_feet = np.array([]).reshape(0, 1, 2)
            else:
                projected_feet = cv2.perspectiveTransform(smoothed_feet, H_smooth)
            
            # Draw visualization
            vis_frame, vis_canvas = draw_visualization(
                frame, court_map, kpts, confs,
                feet_points, projected_feet,
                used_indices, used_dst, stable_side,
                len(tracked_positions)
            )
            
            # Combine frame and canvas side by side
            canvas_resized = cv2.resize(vis_canvas, (COURT_WIDTH, output_height))
            frame_resized = cv2.resize(vis_frame, (frame_width, output_height))
            combined = np.hstack([frame_resized, canvas_resized])
            
            # Write frame
            out.write(combined)
    
    finally:
        cap.release()
        out.release()
        print(f"\nVideo processing complete. Output saved to: {output_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("=" * 80)
    print("Basketball Court Homography Mapping - Advanced Smoothing")
    print("=" * 80)
    
    # Load models
    print("\nLoading models...")
    court_model = YOLO(COURT_MODEL_PATH)
    player_model = YOLO(PLAYER_MODEL_PATH)
    print("✓ Models loaded")
    
    # Load court map
    print("\nLoading court map...")
    court_map = cv2.imread(COURT_MAP_PATH)
    if court_map is None:
        print(f"Error: Could not load court map from {COURT_MAP_PATH}")
        return
    court_map = cv2.resize(court_map, (COURT_WIDTH, COURT_HEIGHT))
    print("✓ Court map loaded")
    
    # Process video
    print("\nStarting video processing...")
    print(f"Smoothing settings:")
    print(f"  - Homography buffer: {HOMOGRAPHY_BUFFER_SIZE} frames")
    print(f"  - Player tracking: {POSITION_BUFFER_SIZE} frames")
    print(f"  - Side detection buffer: {SIDE_BUFFER_SIZE} frames")
    print(f"  - Player retention: {MAX_PLAYER_DISAPPEAR_FRAMES} frames")
    print()
    
    process_video(INPUT_VIDEO, OUTPUT_VIDEO, court_model, player_model, court_map)
    
    print("\n" + "=" * 80)
    print("Processing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()