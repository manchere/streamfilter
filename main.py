import cv2
import mediapipe as mp
import numpy as np

from streamdiffuser import stream_diffusion_frame

# Mediapipe Hands Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

def thermal_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    return cv2.applyColorMap(gray, cv2.COLORMAP_HOT)

def grayscale_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2BGR565)
    return cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
    
def high_contrast_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    high_contrast = clahe.apply(gray)
    return cv2.cvtColor(high_contrast, cv2.COLOR_GRAY2BGR)

filter_functions = {
    1: thermal_filter,
    2: grayscale_filter,
    3: high_contrast_filter
}

def apply_filter(frame, filter_type):
    func = filter_functions.get(filter_type)
    if func:
        return func(frame)
    return frame


def apply_filter_to_shape(frame, filter_type, polygon_points):
    """Apply filter only to the specified polygon region"""
    if polygon_points is None or len(polygon_points) < 3:
        return frame
    
    # Create a mask for the polygon
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    polygon_points = np.array(polygon_points, dtype=np.int32)
    cv2.fillPoly(mask, [polygon_points], 255)
    
    # Apply filter to the entire frame or use streamdiffusion for a special filter_type
    if filter_type == 99:
        # Only process the polygon area with streamdiffusion
        roi = cv2.bitwise_and(frame, frame, mask=mask)
        result_roi = stream_diffusion_frame(roi)
        result = frame.copy()
        result[mask == 255] = result_roi[mask == 255]
        return result
    else:
        filtered_frame = apply_filter(frame, filter_type)
        result = frame.copy()
        result[mask == 255] = filtered_frame[mask == 255]
        return result

def get_thumb_index_points(hand_landmarks, w, h):
    """Get thumb tip and index finger tip points"""
    # Thumb tip (landmark 4)
    thumb_tip = (int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h))
    
    # Index finger tip (landmark 8)
    index_tip = (int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h))
    
    return thumb_tip, index_tip

def create_dynamic_quadrilateral(left_hand_points, right_hand_points):
    """Create a dynamic quadrilateral from two hands' thumb and index finger points"""
    if left_hand_points is None or right_hand_points is None:
        return None
    
    left_hand_thumb, left_hand_index = left_hand_points
    right_hand_thumb, right_hand_index = right_hand_points
    
    points = [left_hand_thumb, left_hand_index, right_hand_index, right_hand_thumb]
    
    return points

def create_single_hand_polygon(hand_points, expansion_factor=50):
    """Create a polygon around single hand's thumb and index finger"""
    if hand_points is None:
        return None
    
    thumb_tip, index_tip = hand_points
    
    # Distance between thumb and index
    distance = np.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)
    
    # Create a dynamic polygon using finger distance
    # When fingers are close, smaller polygon; when far, larger polygon
    expansion = max(20, min(expansion_factor, distance * 0.5))
    
    # Center point
    center_x = (thumb_tip[0] + index_tip[0]) // 2
    center_y = (thumb_tip[1] + index_tip[1]) // 2
    
    # Create rhombus shape around the center
    polygon_points = [
        (center_x, center_y - int(expansion)),  #Top
        (center_x + int(expansion * 0.8), center_y),  #Right
        (center_x, center_y + int(expansion)),  #Bottom
        (center_x - int(expansion * 0.8), center_y)   #Left
    ]
    
    return polygon_points

def check_button_hover(finger_pos, button_positions):
    for i, (bx, by) in enumerate(button_positions):
        if bx-30 < finger_pos[0] < bx+30 and by-30 < finger_pos[1] < by+30:
            return i + 1  # Return filter number (1-5)
    return None

# Button positions
filters = ["diffusion", "retro"]
button_positions = [(175 + i*165, 440) for i in range(2)]

selected_filter = 1  #Start with filter 1
apply_mode = False
polygon_points = None
hand1_points = None
hand2_points = None

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Create a copy for drawing
    display_frame = frame.copy()
    
    # Create tracking visualization area (bottom half)
    tracking_area = np.zeros((h//2, w, 3), dtype=np.uint8)
    
    # Detect hands
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hand_positions = []
    index_positions = []
    detected_hands = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get only thumb and index finger points
            thumb_pos, index_pos = get_thumb_index_points(hand_landmarks, w, h)
            detected_hands.append((thumb_pos, index_pos))
            
            # Store index finger position for button selection
            index_positions.append(index_pos)
            
            # Store hand position for tracking visualization
            hand_positions.append((thumb_pos, index_pos))

    # Update hand points for filtering - ONLY when TWO hands are detected
    if len(detected_hands) >= 2:
        left_hand_points = detected_hands[0]
        right_hand_points = detected_hands[1]
        polygon_points = create_dynamic_quadrilateral(left_hand_points, right_hand_points)
        apply_mode = True
    else:
        # No filter when less than 2 hands
        apply_mode = False
        polygon_points = None
        left_hand_points = None
        right_hand_points = None

    # Button hover and selection
    for finger_pos in index_positions:
        hovered_filter = check_button_hover(finger_pos, button_positions)
        if hovered_filter:
            selected_filter = hovered_filter
            break

    # Draw filter buttons
    for i, (bx, by) in enumerate(button_positions):
        color = (255, 255, 255)  # White for unselected
        if i + 1 == selected_filter:
            color = (0, 255, 0)  # Green for selected
        cv2.rectangle(display_frame, (bx-10, by-30), (bx+130, by+30), color, -1)
        cv2.putText(display_frame, filters[i], (bx + 10, by + 10),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)

    # Apply filter to polygon if active (no visual annotations on main frame)
    if apply_mode and polygon_points:
        display_frame = apply_filter_to_shape(display_frame, selected_filter, polygon_points)

    # Draw tracking visualization (bottom half)
    # Add green dots and lines for tracking visualization
    if hand_positions:
        for hand_pos in hand_positions:
            if hand_pos[0] is not None and hand_pos[1] is not None:
                thumb_pos, index_pos = hand_pos
                # Scale positions for tracking area
                track_x1 = int(thumb_pos[0] * (w / w))
                track_y1 = int((thumb_pos[1] - h//2) * ((h//2) / (h//2))) if thumb_pos[1] > h//2 else 10
                track_x2 = int(index_pos[0] * (w / w))
                track_y2 = int((index_pos[1] - h//2) * ((h//2) / (h//2))) if index_pos[1] > h//2 else 10
                
                # Ensure points are within tracking area bounds
                track_y1 = max(0, min(h//2 - 1, track_y1))
                track_y2 = max(0, min(h//2 - 1, track_y2))
                
                # Draw green dots and lines in tracking area
                cv2.circle(tracking_area, (track_x1, track_y1), 8, (0, 255, 0), -1)
                cv2.circle(tracking_area, (track_x2, track_y2), 8, (0, 255, 0), -1)
                cv2.line(tracking_area, (track_x1, track_y1), (track_x2, track_y2), (0, 255, 0), 3)
                
                # Calculate and display distance
                distance = np.sqrt((thumb_pos[0] - index_pos[0])**2 + (thumb_pos[1] - index_pos[1])**2)
                cv2.putText(tracking_area, f"Distance: {int(distance)}px", 
                           (track_x1 + 20, track_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Combine main frame with tracking area
    merge_frame = np.vstack([display_frame, tracking_area])
    
    # Add status text
    hands_count = len(detected_hands) if detected_hands else 0
    status_text = f"Filter {selected_filter} - {'Active' if apply_mode else 'Inactive'} - Hands: {hands_count}"
    cv2.putText(merge_frame, status_text, (10, h + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if apply_mode and polygon_points:
        if len(detected_hands) == 2:
            cv2.putText(merge_frame, "Two-hand filter active", (10, h + 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Filter box", merge_frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break
    elif key == ord('1'):
        selected_filter = 1
    elif key == ord('2'):
        selected_filter = 2
    elif key == ord('3'):
        selected_filter = 3

cap.release()
cv2.destroyAllWindows()