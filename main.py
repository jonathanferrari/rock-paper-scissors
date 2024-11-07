import cv2
import mediapipe as mp
import math
import random
import time
import numpy as np

def distance(point1, point2):
    """Calculate the Euclidean distance between two landmarks."""
    return math.hypot(point2.x - point1.x, point2.y - point1.y)

def get_finger_status(hand_landmarks):
    """
    Determines whether each finger is open or closed.
    Returns a list of booleans in the order: [thumb, index, middle, ring, pinky]
    """
    # Extract necessary landmarks
    landmarks = hand_landmarks.landmark

    # Thumb: Calculate angle to check if thumb is open
    thumb_tip, thumb_mcp, index_mcp = landmarks[4], landmarks[2], landmarks[5]
    angle = calculate_angle(thumb_tip, thumb_mcp, index_mcp)
    thumb_is_open = angle > 25

    # Finger states for index, middle, ring, and pinky
    finger_status = [thumb_is_open]
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    finger_mcps = [5, 9, 13, 17]

    for tip, pip, mcp in zip(finger_tips, finger_pips, finger_mcps):
        finger_status.append(landmarks[tip].y < landmarks[pip].y < landmarks[mcp].y)

    return finger_status

def calculate_angle(tip, mcp, index_mcp):
    """Calculate angle between thumb tip and index finger MCP for thumb status."""
    x1, y1 = tip.x, tip.y
    x2, y2 = mcp.x, mcp.y
    x3, y3 = index_mcp.x, index_mcp.y
    angle = abs(math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)))
    return angle

def classify_gesture(finger_status):
    """
    Classify hand gesture based on finger status.
    Returns one of ["Rock", "Paper", "Scissors", "Unknown"].
    """
    thumb, index, middle, ring, pinky = finger_status
    if not any(finger_status):
        return "Rock"
    elif all(finger_status):
        return "Paper"
    elif index and middle and not ring and not pinky:
        return "Scissors"
    else:
        return "Unknown"

def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """Overlay img_overlay on top of img at position (x, y) with alpha mask."""
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]

    img_crop[:] = alpha * img_overlay_crop + (1 - alpha) * img_crop

def display_scores(image, user_score, computer_score, tie_score, h):
    """Display game scores at the bottom of the screen."""
    cv2.putText(image, f"Scores - You: {user_score}  Computer: {computer_score}  Ties: {tie_score}", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

def determine_winner(user_choice, computer_choice):
    """Determine the game result and update the score based on choices."""
    if user_choice == computer_choice:
        return "Tie!"
    elif (user_choice == "Rock" and computer_choice == "Scissors") or \
         (user_choice == "Paper" and computer_choice == "Rock") or \
         (user_choice == "Scissors" and computer_choice == "Paper"):
        return "You Win!"
    else:
        return "Computer Wins!"

def main():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.7, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)

    # Load and resize gesture images
    gesture_images = {opt: cv2.resize(cv2.imread(f'img/{opt.lower()}.png', cv2.IMREAD_UNCHANGED), (200, 200)) 
                      for opt in ["Rock", "Paper", "Scissors"]}

    # Initialize game state
    user_score, computer_score, tie_score, round_number = 0, 0, 0, 0
    game_state, gesture_start_time, detected_gesture = "waiting", None, None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        h, w, c = image.shape

        if game_state == "waiting":
            cv2.putText(image, "Show your gesture: Rock, Paper, or Scissors", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                finger_status = get_finger_status(hand_landmarks)
                gesture = classify_gesture(finger_status)

                cx, cy = int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h)
                cv2.putText(image, gesture, (cx - 30, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

                if gesture in gesture_images:
                    if detected_gesture != gesture:
                        detected_gesture, gesture_start_time = gesture, time.time()
                    elif time.time() - gesture_start_time >= 1:
                        user_choice, computer_choice = gesture, random.choice(["Rock", "Paper", "Scissors"])
                        result = determine_winner(user_choice, computer_choice)
                        
                        if result == "You Win!": user_score += 1
                        elif result == "Computer Wins!": computer_score += 1
                        else: tie_score += 1

                        round_number += 1
                        game_state, result_start_time = "result", time.time()
                        detected_gesture, gesture_start_time = None, None
            else:
                detected_gesture, gesture_start_time = None, None

            display_scores(image, user_score, computer_score, tie_score, h)

        elif game_state == "result":
            image = np.zeros_like(image)
            display_result_screen(image, w, h, gesture_images, user_choice, computer_choice, result, round_number,
                                  user_score, computer_score, tie_score)

            if time.time() - result_start_time >= 2:
                game_state = "waiting"

        cv2.imshow('Rock Paper Scissors', image)
        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]: break

    cap.release()
    cv2.destroyAllWindows()

def display_result_screen(image, w, h, gesture_images, user_choice, computer_choice, result, round_number,
                          user_score, computer_score, tie_score):
    """Display round result with player and computer gestures."""
    cv2.putText(image, f"Round {round_number}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    overlay_image_alpha(image, gesture_images[user_choice][:, :, :3], 50, 100, gesture_images[user_choice][:, :, 3] / 255.0)
    overlay_image_alpha(image, gesture_images[computer_choice][:, :, :3], w - 250, 100, gesture_images[computer_choice][:, :, 3] / 255.0)
    cv2.putText(image, result, (int(w / 2) - 100, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 4)
    display_scores(image, user_score, computer_score, tie_score, h)

if __name__ == '__main__':
    main()