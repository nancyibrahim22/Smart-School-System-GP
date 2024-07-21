import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import os
import time
import re
import SQLServerDoWork
import datetime
import joblib
names_negative_list = []
pos_list_own_images = np.array([])
neg_list_own_images = np.array([])
positive = []
negative = []
positive_distance_list = []
names_positive_list = []
negative1=[]
threshold = 1.5
pos_sum = 0
neg_sum = 0

encoder = tf.keras.models.load_model(r'D:\Hena_Version\SchoolFinal1\VGG16encoder_lfw_casia_without_aug_delete1_max4_100epochs_early_printtrain.h5')
# Load your pre-trained liveness detection model
liveness_model = tf.keras.models.load_model(r'D:\Hena_Version\SchoolFinal1\best_liveness_model_1.h5')
def FaceYacta():
    # function 3shan ashof el liveness
    def is_live_face(face_image):
        # Preprocess the face image
        face_image = cv2.resize(face_image, (128, 128))
        face_image = face_image.astype('float32') / 255.0
        # face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        # face_image = np.array(Image.fromarray(face_image).resize((128, 128)))
        face_image = np.expand_dims(face_image, axis=0)

        # Predict liveness
        prediction = liveness_model.predict(face_image)
        # if prediction[0][0] > 0.1:
        #     return "Live"
        # else:
        #     return "Spoof"
        print('live threshold: \t', prediction[0][0])
        return prediction[0][0] > 0.2

    def read_our_image(path):
        print(path)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(Image.fromarray(image).resize((128, 128)))

        face_cascade = cv2.CascadeClassifier(r'D:\Hena_Version\SchoolFinal1\haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5)
        try:
            x, y, w, h = faces[0]
            face_img = image[y:y + h, x:x + w]
            face_img = np.array(Image.fromarray(face_img).resize((128, 128)))

        except:
            face_img = image

        return face_img

    def read_Anchor(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(Image.fromarray(image).resize((128, 128)))

        face_cascade = cv2.CascadeClassifier(r'D:\Hena_Version\SchoolFinal1\haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        try:
            x, y, w, h = faces[0]
            face_img = image[y:y + h, x:x + w]
            face_img = np.array(Image.fromarray(face_img).resize((128, 128)))
        except:
            face_img = image

        return face_img



    def my_model():
        name="Lesa Mt3mlsh"
        global names_negative_list
        global negative
        global negative1
        our_images_path = r'D:\Hena_Version\SchoolFinal1\1 Shot'
        folders = os.listdir(our_images_path)
        for folder in folders:
            folder_path = os.path.join(our_images_path, folder)
            print("folder  ", folder_path)
            for filename in os.listdir(folder_path):
                negative_path = os.path.join(folder_path, filename)
                names_negative_list = np.append(names_negative_list, filename)
                image = read_our_image(negative_path)
                image = np.expand_dims(image, axis=0)
                image = np.array(image)
                image = preprocess_input(image)
                negative.append(image)

        for i in range (len(negative)):
            negative1.append(encoder.predict(negative[i]))

    def DoWork(face_img):
        negative_distance_list = []
        global names_negative_list
        global neg_list_own_images
        global neg_sum
        distances = []
        final_names_avg = []
        global negative1
        anchor = []
        anchor.append(read_Anchor(face_img))
        anchor = np.array(anchor)
        anchor = preprocess_input(anchor)
        tensor1 = encoder.predict(anchor)
        max_counter = 0

        for i in negative1:
            tensor3=i
            neg_distance = np.sum(np.square(tensor1 - tensor3), axis=-1)
            negative_distance_list = np.append(negative_distance_list, neg_distance)
            neg_prediction = np.where(neg_distance <= threshold, 0, 1)
            neg_list_own_images = np.append(neg_list_own_images, neg_prediction)
        for i in range(len(negative_distance_list)):
            neg_sum += negative_distance_list[i]
            if (i + 1) % 1 == 0:
                neg_avg = neg_sum / 1
                distances = np.append(distances, neg_avg)
                final_names_avg = np.append(final_names_avg, names_negative_list[i])
                neg_sum = 0

        min_distance_avg = min(distances)
        for i in range(len(distances)):
            if min_distance_avg == distances[i]:
                min_name = final_names_avg[i]
                break

        print(final_names_avg)
        print(distances)
        print(min_distance_avg)
        if min_distance_avg>threshold:
            return "unknown"
        print(min_name)
        return min_name



    # Initialize Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(r'D:\Hena_Version\SchoolFinal1\haarcascade_frontalface_default.xml')

    if face_cascade.empty():
        print("Error: Could not load Haar Cascade Classifier.")
        exit()

    # Initialize the camera
    cap = cv2.VideoCapture(0)

    my_model()

    # Flag to indicate if a photo has been captured
    photo_captured = False


    # Flag to indicate if a face is currently detected
    face_detected = False
    count=0
    faces_list_capturing=[]
    names_list_capturing = []
    Times_list_capturing = []
    paths_list_capturing = []



    # Load the pre-trained clustering model
    model_path = r'D:\Hena_Version\SchoolFinal1\uniform_model.joblib'
    clustering_model = joblib.load(model_path)
    # Define the black cluster index (assuming you know which cluster index corresponds to black)
    black_cluster_index = 'white'  # Change this to your actual black cluster index

    # Initialize the video capture
    cap = cv2.VideoCapture(0)

    # Load YOLO model
    net = cv2.dnn.readNetFromDarknet(r'D:\Hena_Version\SchoolFinal1\yolov3.cfg',
                                     r'D:\Hena_Version\SchoolFinal1\yolov3.weights')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Load the COCO class names
    with open('D:\Hena_Version\SchoolFinal1\coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    unconnected_out_layers = net.getUnconnectedOutLayers()

    # Handle both cases of `net.getUnconnectedOutLayers()`
    if isinstance(unconnected_out_layers[0], list) or isinstance(unconnected_out_layers[0], np.ndarray):
        output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]
    else:
        output_layers = [layer_names[i - 1] for i in unconnected_out_layers]

    # output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def extract_color_histogram(image, bins=(8, 8, 8)):
        # Convert the image to the HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Compute the color histogram
        hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
        # Normalize the histogram
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def face_uni(image,count):
        # Prepare the image for YOLO
        image = cv2.imread(image)
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Initialize lists to hold detection details
        class_ids = []
        confidences = []
        boxes = []
        check_uniform = False
        # Process each detection
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id] == 'person':  # Filter for persons
                    center_x = int(detection[0] * image.shape[1])
                    center_y = int(detection[1] * image.shape[0])
                    w = int(detection[2] * image.shape[1])
                    h = int(detection[3] * image.shape[0])
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # for i in range(len(boxes)):
        #     if i in indexes:
        #         x, y, w, h = boxes[i]
        if len(indexes) > 0:
            # Find the maximum rectangle
            max_area = 0
            max_index = 0
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                area = w * h
                if area > max_area:
                    max_area = area
                    max_index = i

            # Use the maximum rectangle
            x, y, w, h = boxes[max_index]
            # Adjust the ROI to focus on the upper body
            upper_body_y = int(y + h * 0.45)  # Adjust as needed
            upper_body_h = int(h * 0.55)  # Adjust as needed
            # Ensure the bounding box is within the frame's bounds
            if upper_body_y >= 0 and upper_body_y + upper_body_h <= image.shape[0] and x >= 0 and x + w <= image.shape[
                1]:
                upper_body = image[upper_body_y:upper_body_y + upper_body_h, x:x + w]

                if upper_body.size > 0:  # Ensure the cropped image is valid
                    # Extract color features from the upper body region
                    color_features = extract_color_histogram(upper_body).reshape(1, -1)

                    # Predict the cluster
                    cluster = clustering_model.predict(color_features)
                    print(cluster, '\n')
                    # Check if it belongs to the black cluster
                    if cluster == black_cluster_index:
                        label = "Uniform: Yes"
                        check_uniform = True
                        color = (0, 255, 0)  # Green
                    else:
                        label = "Uniform: No"
                        check_uniform = False
                        color = (0, 0, 255)  # Red
                    # Draw a rectangle around the detected upper body
                    cv2.rectangle(image, (x, upper_body_y), (x + w, upper_body_y + upper_body_h), color, 2)
                    cv2.putText(image, label, (x, upper_body_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    photo_name = f'D:\Hena_Version\SchoolFinal1\captured_photo_uniform{count}.jpg'
                    # el frame da bnhotlo esm el sora (photo_name)
                    cv2.imwrite(photo_name, image)
            else:
                print("Bounding box is out of bounds.")
                # Handle the case where the bounding box is out of bounds
                # Log the event
                # with open("bounding_box_errors.log", "a") as log_file:
                #     log_file.write(f"Bounding box out of bounds: x={x}, y={y}, w={w}, h={h}\n")

                # Optionally, notify the user
                cv2.putText(image, "Bounding box out of bounds", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                photo_name = f'D:\Hena_Version\SchoolFinal1\error_photo{count}.jpg'
                cv2.imwrite(photo_name, image)
        return photo_name, check_uniform


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        time.sleep(0.4)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        print(len(faces))
        print('-------')
        if count==0: #Stpoed here

            faces_list_capturing=np.append(faces_list_capturing,len(faces))
            if len(faces) > 0:
                #count += 1
                # Face detected
                if not face_detected:

                    for (x, y, w, h) in faces:
                            # Preprocess the detected face (resize, normalize, etc.)
                            face_img = frame[y:y+h, x:x+w]
                            live = is_live_face(frame)
                            print('liveeee: \t', live)
                            if live:
                                count += 1
                                student_name=DoWork(face_img)
                                # Display the result on the frame
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                                text_box = student_name.split("_")[0]


                                # Input string
                                input_string = text_box

                                # Use regular expression to extract non-numeric part
                                result = re.match("([a-zA-Z]+)", input_string)

                                if result:
                                    extracted_string = result.group(1)
                                else:
                                    extracted_string = text_box
                                cv2.putText(frame, extracted_string, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                                # Capture a photo when a face is detected

                                photo_name = f'D:\Hena_Version\SchoolFinal1\captured_photo_{count}.jpg'
                                cv2.imwrite(photo_name, frame)
                                if student_name=="unknown":
                                    pass
                                else:
                                    number = int(student_name.split("_")[1].split(".")[0])
                                    if len(names_list_capturing) == 0:
                                        current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                        names_list_capturing = np.append(names_list_capturing, number)
                                        Times_list_capturing = np.append(Times_list_capturing, current_datetime)

                                        paths_list_capturing = np.append(paths_list_capturing, photo_name)
                                    else:
                                        if number in names_list_capturing:
                                            pass
                                        else:
                                            names_list_capturing = np.append(names_list_capturing, number)
                                            current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                            Times_list_capturing = np.append(Times_list_capturing, current_datetime)
                                            paths_list_capturing = np.append(paths_list_capturing, photo_name)
                                photo_captured = True
                            #break
            else:
                # No face detected
                face_detected = False
        else:
            #print('hahahah')
            #print(faces_list_capturing)
            if len(faces) > 0 and faces_list_capturing[-1]==0:

                faces_list_capturing = np.append(faces_list_capturing, len(faces))
                #count += 1
                # Face detected
                if not face_detected:
                    # Capture a photo when a new face is detected
                    # cv2.imwrite(f'captured_photo_{count}.jpg', frame)
                    # face_detected = True
                    for (x, y, w, h) in faces:
                        # Preprocess the detected face (resize, normalize, etc.)
                        face_img = frame[y:y + h, x:x + w]
                        live = is_live_face(frame)
                        if live:
                            count+=1
                            student_name = DoWork(face_img)
                            # Display the result on the frame
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            text_box = student_name.split("_")[0]
                            # Input string
                            input_string = text_box

                        # Use regular expression to extract non-numeric part
                            result = re.match("([a-zA-Z]+)", input_string)

                            if result:
                                extracted_string = result.group(1)
                            else:
                                extracted_string = text_box
                            cv2.putText(frame, extracted_string, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                            photo_name = f'D:\Hena_Version\SchoolFinal1\captured_photo_{count}.jpg'
                        # Capture a photo when a face is detected
                            cv2.imwrite(photo_name, frame)

                            if student_name == "unknown":
                                pass
                            ###########
                            else:
                                number = int(student_name.split("_")[1].split(".")[0])
                                if len(names_list_capturing) == 0:
                                    names_list_capturing = np.append(names_list_capturing, number)
                                    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    Times_list_capturing = np.append(Times_list_capturing, current_datetime)
                                    paths_list_capturing = np.append(paths_list_capturing, photo_name)
                                else:
                                    if number in names_list_capturing:
                                        pass
                                    else:
                                        names_list_capturing = np.append(names_list_capturing, number)
                                        current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                        Times_list_capturing = np.append(Times_list_capturing, current_datetime)
                                        paths_list_capturing = np.append(paths_list_capturing, photo_name)
                            photo_captured = True
                        # break
            else:
                faces_list_capturing = np.append(faces_list_capturing, len(faces))
                # No face detected
                face_detected = False

        # Display the result on the frame
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the frame
        cv2.imshow('Face Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(names_list_capturing)
    print(paths_list_capturing)
    SQLServerDoWork.AddStudentAttendance(names_list_capturing,Times_list_capturing)
    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()
    count=0
    uniform_paths = []
    uniform_bool_paths = []
    for captured_path in paths_list_capturing:
        count+=1
        retured_uniform, returned_bool = face_uni(captured_path,count)
        uniform_paths = np.append(uniform_paths, retured_uniform)
        uniform_bool_paths = np.append(uniform_bool_paths, returned_bool)
    print(uniform_paths)
    print(uniform_bool_paths)

#FaceYacta()