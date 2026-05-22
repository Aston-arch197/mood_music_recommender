from deepface import DeepFace
import cv2
import pandas as pd
import random
import webbrowser 
import time 

# Load songs dataset
songs_df = pd.read_csv("songs.csv")

# Open webcam
cap = cv2.VideoCapture(0)

print("Press R to detect mood and recommend song")
print("Press Q to quit")

# Variables to store current recommendation
current_song = ""
current_artist = ""
current_emotion = ""

recent_songs = []

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Show emotion on screen
    cv2.putText(
        frame,
        f"Emotion: {current_emotion}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show recommended song
    cv2.putText(
        frame,
        f"{current_song} - {current_artist}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Show webcam window
    cv2.imshow("Mood Music Recommender", frame)

    # Key controls
    key = cv2.waitKey(1) & 0xFF

    # Press R to analyze emotion
    if key == ord('r'):

        try:
            # Analyze emotion
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False
            )

            # Get dominant emotion
            emotion = result[0]['dominant_emotion']
            current_emotion = emotion

            # Filter songs matching emotion
            filtered_songs = songs_df[songs_df['mood'] == emotion]

            # Random recommendation
            if not filtered_songs.empty:
                # Remove recently played songs
                available_songs = filtered_songs[
                 ~filtered_songs['song'].isin(recent_songs)
                ]

# If all songs were recently played, reset
                if available_songs.empty:
                    recent_songs.clear()
                    available_songs = filtered_songs

# Pick random song
                recommended_song = available_songs.sample(1).iloc[0]
                current_song = recommended_song['song']
                recent_songs.append(current_song)

# Keep only last 2 songs
                if len(recent_songs) > 2:
                    recent_songs.pop(0)
                current_artist = recommended_song['artist']

                spotify_link = recommended_song['spotify_link']

                print(f"\nDetected Emotion: {emotion}")
                print(f"Recommended Song: {current_song} by {current_artist}")

                cv2.imshow("Mood Music Recommender", frame)
                cv2.waitKey(3000)  # wait 3 seconds

                #open spotify using browser 
                webbrowser.open(spotify_link)

            else:
                print("No matching songs found.")

        except Exception as e:
            print("Error:", e)

    # Press Q to quit
    if key == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()