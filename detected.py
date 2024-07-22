#importation des bibliotheques
import cv2 #cv2 est la bibliothèque OpenCV, qui est utilisée pour le traitement d'images et de vidéos
import streamlit as st  # pour les applicayions web en python 

#Le classificateur de cascade de visages est un modèle pré-entraîné qui peut être utilisé pour détecter des visages dans des images et des vidéos

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#Créer une fonction pour capturer des images de la webcam et détecter les visages

def detect_faces():
    # initialisation de la web came
    cap = cv2.VideoCapture(0)
    while True:
        #lecture de l'image
        ret, frame = cap.read()
        # convertion en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detecter le viage
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Display the frames
        cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)
        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
# application streamlit 
def app():
    st.title("Face Detection using Viola-Jones Algorithm")
    st.write("Press the button below to start detecting faces from your webcam")
    # ajouter un bouton pour la detection de visage
    if st.button("Detect Faces"):
        # appel la fonction de dectection
        detect_faces()
if __name__ == "__main__":
    app()