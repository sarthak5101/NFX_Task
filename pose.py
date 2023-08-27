import cv2
import mediapipe as mp

def main(source):
    # Initialize MediaPipe drawing utilities and Pose model
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Open the video capture source
    cap = cv2.VideoCapture(source)

    while cap.isOpened():
        try:
            # Read frame from the video capture
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame for pose detection
            pose_results = pose.process(frame_rgb)

            # Draw skeleton on the frame
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display the frame with pose landmarks
            cv2.imshow('Output', frame)
            cv2.waitKey(1)  # Add a small delay for the frame to be displayed

        except Exception as e:
            print("An error occurred:", e)
            break

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'video.mp4'
    webcam_source = 0
    
    # Call the main function with the appropriate source
    main(video_path)  # For video file
    # main(webcam_source)  # For webcam