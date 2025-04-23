# InfintySolve

A Django-based AI-powered web application that enables users to **solve handwritten math problems using hand gestures** and real-time webcam input. This innovative system uses **computer vision**, **Gemini AI (Google Generative AI)**, and **hand tracking** to provide an interactive gesture-based calculator experience.

## 🚀 Features

- ✍️ Draw math equations in the air using hand gestures.
- 🤖 Real-time gesture recognition and smooth drawing canvas.
- 📷 Live webcam feed using OpenCV and CVZone.
- 🔍 Solve handwritten problems using Google Gemini AI.
- 📤 Upload images of handwritten problems to get instant solutions.
- 🔄 Streamlined user interface with Django and HTML templates.

## 🛠️ Tech Stack

- **Backend**: Django, Python
- **AI Integration**: Google Generative AI (Gemini)
- **Computer Vision**: OpenCV, CVZone (HandTrackingModule)
- **Image Processing**: PIL (Pillow)
- **Frontend**: HTML, StreamingHttpResponse

## **📸 How It Works**

Gesture Input: Uses index finger to draw on a virtual canvas.

Canvas Analysis: The drawn image is captured and sent to Gemini AI.

AI Response: The model returns a step-by-step solution.

Image Upload: Optionally, users can upload an image to get an instant answer.

## **✍️ Hand Gestures**

☝️ Index Finger → Draw

🖐️ All Fingers → Clear canvas

👍 Thumb Only → Submit to AI

✌️ Index + Middle → Pause drawing

## **🧠 Future Improvements**

Add support for more complex math (graphs, equations).

Enhance drawing smoothness using better trajectory prediction.

Deploy on cloud (e.g., AWS, GCP) for public access.
