import cv2

def stream_video(source=0):
    """
    Streams video frames from the specified source.

    Args:
        source (int or str): If an integer is provided, it will use the webcam at that index. 
                             If a string is provided, it will use the specified video file.

    Yields:
        frame: The current video frame.
    """
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Unable to open video source {source}")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        yield frame

        # Check if 'q' or 'Esc' is pressed to exit
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()



def main():
    # Example usage:
    for frame in stream_video(0):  # Change 0 to 'video.mp4' for a video file
        # Example: Display the frame (you can also process or save it instead)
        cv2.imshow('Video Stream', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()