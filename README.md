# Skin-oiliness-severity-Detection
#Libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the path to your dataset
train_data_dir = 'file path'
test_data_dir = 'file path'
valid_data_dir = 'file path'

# Define image size and batch size
img_size = (224, 224)
batch_size = 32

# Data generators with data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    valid_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

#CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))  # Assuming 4 oiliness levels

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=valid_generator
)

# Evaluation of the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc}')

# Prediction on the test set
predictions = model.predict(test_generator)
predicted_classes = tf.argmax(predictions, axis=1)

# Getting true labels
true_labels = test_generator.classes

# Classification report
from sklearn.metrics import classification_report
print(classification_report(true_labels, predicted_classes))

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

#Prediction of oiliness and displaying level of Oiliness
def predict_and_display(image_path):

    img = image.load_img(image_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0


    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Mapping class to oiliness severity levels (1 to 4)
    oiliness_severity = {0: 'Severity 1', 1: 'Severity 2', 2: 'Severity 3', 3: 'Severity 4'}

    # Displaying the image and predicted oiliness severity
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Predicted Oiliness Severity: {oiliness_severity[predicted_class]}')
    plt.show()


    print(f'The predicted oiliness severity for {image_path} is: {oiliness_severity[predicted_class]}')

image_path = 'image path'

predict_and_display(image_path)

#Test Data Set (Detecting Oiliness for 10 Images)
test_data_dir = '/content/drive/MyDrive/Skin Detection/Attempt 2/Skin Types/test/oily'

import os
# Get the list of images in the test set
test_image_list = os.listdir(test_data_dir)
test_image_list = [os.path.join(test_data_dir, img) for img in test_image_list]

# Selection of the first 10 images
first_10_images = test_image_list[:10]

#Prediction of oiliness and displaying level of Oiliness
def predict_and_display(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)


    oiliness_severity = {0: 'Severity 1', 1: 'Severity 2', 2: 'Severity 3', 3: 'Severity 4'}


    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Predicted Oiliness Severity: {oiliness_severity[predicted_class]}')
    plt.show()


    print(f'The predicted oiliness severity for {image_path} is: {oiliness_severity[predicted_class]}')

for image_path in first_10_images:
    predict_and_display(image_path)

import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from tensorflow.keras.preprocessing import image

def predict_oiliness(img_array):
    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)


    oiliness_severity = {0: 'Severity 1', 1: 'Severity 2', 2: 'Severity 3', 3: 'Severity 4'}


    return oiliness_severity[predicted_class]


    # Function to detect skin regions
def detect_skin(image, lower_bound, upper_bound):
    # Convert image to YCrCb color space
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

# Function to detect skin regions
def detect_skin(image, lower_bound, upper_bound):
    # Convert image to YCrCb color space
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Mask skin color
    skin_mask = cv2.inRange(ycrcb, lower_bound, upper_bound)

    # Apply morphology to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)

    # Apply mask to image
    skin_detected = cv2.bitwise_and(image, image, mask=skin_mask)

    return skin_detected, skin_mask

# Function to process webcam feed
def process_webcam():
    cap = cv2.VideoCapture(0)

    # Define initial skin color thresholds
    lower_bound = np.array([0, 133, 77], dtype=np.uint8)
    upper_bound = np.array([255, 173, 127], dtype=np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect skin regions
        skin_detected, skin_mask = detect_skin(frame, lower_bound, upper_bound)

        # Preprocess the skin-detected image for oiliness prediction
        img_array = cv2.resize(skin_detected, img_size)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Make prediction and get oiliness severity
        oiliness_severity = predict_oiliness(img_array)

        # Display the result
        cv2.putText(skin_detected, f'Oiliness: {oiliness_severity}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2_imshow(skin_detected)

        # Check for key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Call the function to process webcam feed
process_webcam()

#Javascript for Capturing Images

from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

def take_photo(filename='photo.jpg', quality=0.8):
    js = Javascript('''
        async function takePhoto(quality) {
            const div = document.createElement('div');
            const capture = document.createElement('button');
            capture.textContent = 'Capture';
            div.appendChild(capture);

            const video = document.createElement('video');
            video.style.display = 'block';
            const stream = await navigator.mediaDevices.getUserMedia({ 'video': true });

            document.body.appendChild(div);
            div.appendChild(video);
            video.srcObject = stream;
            await video.play();

            // Resize the output to fit the video element.
            google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

            // Wait for Capture to be clicked.
            await new Promise((resolve) => capture.onclick = resolve);

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            stream.getVideoTracks()[0].stop();
            div.remove();
            return canvas.toDataURL('image/jpeg', quality);
        }
    ''')
    display(js)

    # Capture the image using JavaScript
    data = eval_js('takePhoto({})'.format(quality))

    # Decode the image data
    binary = b64decode(data.split(',')[1])

    # Save the image
    with open(filename, 'wb') as f:
        f.write(binary)
    return filename

# Capture an image from the webcam
image_path = take_photo()
print('Image captured and saved as:', image_path)

predict_and_display(image_path)

