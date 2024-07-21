import time
import cv2
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import re
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Load the saved Siamese model
model = tf.keras.models.load_model(r'mobilenetv2_ourdata_lfw_without_aug_3.h5')
# Load your pre-trained liveness detection model
liveness_model = tf.keras.models.load_model(r'best_liveness_model.h5')

#function 3shan ashof el liveness
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
    return prediction[0][0] > 0.4


def read_our_image(path):
    print(path)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(Image.fromarray(image).resize((128, 128)))
    # Load YOLO model
    net = cv2.dnn.readNet(r'yolov3-wider_16000.weights',
                          r'face.cfg')
    layer_names = net.getLayerNames()
    unconnected_out_layers = net.getUnconnectedOutLayers()
    # Handle both cases of `net.getUnconnectedOutLayers()`
    if isinstance(unconnected_out_layers[0], list) or isinstance(unconnected_out_layers[0], np.ndarray):
        output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]
    else:
        output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
    # output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    conf_threshold = 0.5
    nms_threshold = 0.3
    height, width, channels = image.shape

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process YOLO output
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold and class_id == 0:  # Check for class 'person'
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    if (len(boxes) > 0 and len(indexes)>0):
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                face_img = image[y:y + h, x:x + w]
                # Ensure the bounding box coordinates are within the frame dimensions
                if x < 0: x = 0
                if y < 0: y = 0
                if x + w > image.shape[1]: w = image.shape[1] - x
                if y + h > image.shape[0]: h = image.shape[0] - y

                face_img = image[y:y + h, x:x + w]

                # Check if face_img is not empty
                if face_img.size != 0:
                    face_img = np.array(Image.fromarray(face_img).resize((128, 128)))
    return face_img

'''dy function b bdeha el image aly hya el ive photo w b read el image bnfs el tare2a aly b read beha el images 
fy el code bta3 el model el far2 mafesh read image 3shan ana msh badeha path w ttl3 mno el sora la bdeha el sora 
3ala tol w b extract el face using haarcascade
'''
def read_Anchor(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(Image.fromarray(image).resize((128, 128)))
    # Load YOLO model
    net = cv2.dnn.readNet(r'yolov3-wider_16000.weights',
                          r'face.cfg')
    layer_names = net.getLayerNames()
    unconnected_out_layers = net.getUnconnectedOutLayers()
    # Handle both cases of `net.getUnconnectedOutLayers()`
    if isinstance(unconnected_out_layers[0], list) or isinstance(unconnected_out_layers[0], np.ndarray):
        output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]
    else:
        output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
    # output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    conf_threshold = 0.5
    nms_threshold = 0.3
    height, width, channels = image.shape

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process YOLO output
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold and class_id == 0:  # Check for class 'person'
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    if(len(boxes)>0 and len(indexes)>0):
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                face_img = image[y:y + h, x:x + w]
                # Ensure the bounding box coordinates are within the frame dimensions
                if x < 0: x = 0
                if y < 0: y = 0
                if x + w > image.shape[1]: w = image.shape[1] - x
                if y + h > image.shape[0]: h = image.shape[0] - y

                face_img = image[y:y + h, x:x + w]

                # Check if face_img is not empty
                if face_img.size != 0:
                    face_img = np.array(Image.fromarray(face_img).resize((128, 128)))
    else:
        face_img = image
    return face_img

#array feha asamy kol el files
names_negative_list = []

#array
neg_list_own_images = np.array([])

#el array aly hyb2a feha el sewar kolha b3d el preprocessing
negative = []

#el sewar kolha b3d ma at3mlha encoding
negative1=[]

'''el threshold aly hoa el distance ma ben el sora w ell anchor lw a2l mn 
el threshold da or equals yb2a (0) positive w lw akbr yb2a (1) negative'''
threshold = 1.5

#el sum bta3 el distances bta3 kol sora m3 el anchor fy folder wahed
neg_sum = 0

'''
function bt3dy 3ala kol el folders w el files kolha w bt3ml read ll files dy w nafs el preprocessing 
aly mawgod fy el code bta3 el model w b3den y3mllhom encoding kolhom w yhothom fy array esmha negative1
'''
def my_model():
    name="Lesa Mt3mlsh"
    global names_negative_list
    global negative
    global negative1
    our_images_path = r'1 Shot'
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
        negative1.append(model.predict(negative[i]))

'''
function btakhod parameter aly hya el sora el live (anchor) w b3den by3mllha read w preprocessing w encoding
f byb2a m3ana el encoding bta3 el anchor fy (tensor1) w b3den by3dy 3ala el array aly hoa (negative1) aly feh
el encoding bta3 kol el sewar w yhot kol wahda fy (tensor3) w b3den yhsb el distance (neg_distance) 
w b3den y append kol el distances bta3t kol sora m3 el anchor fy array (negative_distance_list) w yshof el distance dy
m3 el thershold htb2a (0) positive wala (1) negative w el result bttht fy (neg_prediction) w 
y3ml append l kol sora hya 0 or 1 fy array (neg_list_own_images).......
w b3den bnlf 3ala el array (negative_distance_list) aly feha kol el distances w 
ngeb el sum bta3 el distances bta3 kol sora m3 el anchor fy folder wahed w nhoto fy (neg_sum) 
w b3den ngeb el average w nhoto fy (neg_avg) w b3den n append kol average bta3 folder fy array (distances)
w b3den kol average bta3 folder b3rffo bs hoa da bta3 anhy folder an bakhod ay esm file gwa el folder da 
w b3ml append fy el array (final_names_avg)....
w b3den bgeb minimum value fy el array (distances) aly feh kol el average bta3 el folders w bhoto fy (min_distance_avg) 
w b3den blf 3ala el array da w bshof el minimum da kan bta3 anhy shakhs f bgebo mn el array da (final_names_avg) w bhoto fy (min_name)......
w b3den b check min_distance_avg akbr mn el threshold wala la lw akbr yb2a da el anchor unknown 
lw la yb2a da (min_name) w b return unknown aw el esm aly fy el (min_name)
'''
def DoWork(face_img):
    #array feha kol el distances bta3t el sewar m3 el anchor
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
    tensor1 = model.predict(anchor)
    max_counter = 0

    #loop btgeb el distances kol sora m3 el anchor w nshof m3 el thershold hya positive wala negative
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

# bnshghl el camera
# cap = cv2.VideoCapture(0)

#bn3ml encoding l kol el sewar aly fy el folders
# my_model()

# Flag to indicate if a photo has been captured
photo_captured = False

# Flag to indicate if a face is currently detected
face_detected = False

#counter ll sewar kol sora hyt3mlha capture leha number f hyb2a el counter da
count=0

#array feh 3andy kam face odam el camera now
faces_list_capturing=[]

#array feha el IDs bta3t el nas aly at3mllhom detect abl keda
names_list_capturing = []

#array feha el path bta3 el sora bta3 el shakhs aly hadar
paths_list_capturing = []



# Load YOLO model
net = cv2.dnn.readNet(r'yolov3-wider_16000.weights', r'face.cfg')
layer_names = net.getLayerNames()
unconnected_out_layers = net.getUnconnectedOutLayers()
# Handle both cases of `net.getUnconnectedOutLayers()`
if isinstance(unconnected_out_layers[0], list) or isinstance(unconnected_out_layers[0], np.ndarray):
    output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]
else:
    output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
#output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
conf_threshold = 0.5
nms_threshold = 0.3

# Initialize the camera
cap = cv2.VideoCapture(0)
my_model()


while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Perform face detection on the frame
    #time.sleep(2)
    height, width, channels = frame.shape

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process YOLO output
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold and class_id == 0:  # Check for class 'person'
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    if count==0:
        faces_list_capturing=np.append(faces_list_capturing,len(indexes))
        if len(indexes) > 0:
            # Face detected
            if not face_detected:
                if(len(boxes)>0):
                    for i in range(len(boxes)):
                        if i in indexes:
                            x, y, w, h = boxes[i]

                            live = is_live_face(frame)
                            print('liveeee: \t', live)
                            if live:
                                count += 1
                                face_img = frame[y:y + h, x:x + w]
                                # Ensure the bounding box coordinates are within the frame dimensions
                                if x < 0: x = 0
                                if y < 0: y = 0
                                if x + w > frame.shape[1]: w = frame.shape[1] - x
                                if y + h > frame.shape[0]: h = frame.shape[0] - y

                                face_img = frame[y:y + h, x:x + w]

                                # Check if face_img is not empty
                                if face_img.size != 0:
                                    student_name = DoWork(face_img)

                                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                                    text_box = student_name.split("_")[0]
                                    # Input string
                                    input_string = text_box

                                    # Use regular expression to extract non-numeric part (byshel ay raqam fy el text bb2a 3ayza el esm bs)
                                    result = re.match("([a-zA-Z]+)", input_string)
                                    # If a match is found, result.group(1) retrieves the part of the string that matched the first capturing group in the pattern (the entire [a-zA-Z]+ part).
                                    if result:
                                        extracted_string = result.group(1)
                                    else:
                                        extracted_string = text_box
                                    '''el text aly byb2a mktob tht el rectangle aly fy el sora hyb2a student name bal el (_)
                                        3shan esm al shakhs byb2a abl el (_) w b3do byb2a el ID bta3 el shakhs'''
                                    cv2.putText(frame, extracted_string, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                                (255, 0, 0), 2)

                                    # Capture a photo when a face is detected
                                    photo_name = f'captured_photo_hhh{count}.jpg'
                                    # el frame da bnhotlo esm el sora (photo_name)
                                    cv2.imwrite(photo_name, frame)
                                    if student_name == "unknown":
                                        pass
                                    else:
                                        # extract el student ID mn el filename w bhoto fy (number)
                                        number = int(student_name.split("_")[1].split(".")[0])
                                        if len(names_list_capturing) == 0:
                                            names_list_capturing = np.append(names_list_capturing, number)
                                            paths_list_capturing = np.append(paths_list_capturing, photo_name)
                                        else:
                                            if number in names_list_capturing:
                                                pass
                                            else:
                                                names_list_capturing = np.append(names_list_capturing, number)
                                                paths_list_capturing = np.append(paths_list_capturing, photo_name)
                                    photo_captured = True
        else:
            # No face detected (el length bta3 (faces)=0) mafesh face delw2ty
            face_detected = False
    else:
        if len(indexes) > 0 and faces_list_capturing[-1]!=0:
            faces_list_capturing = np.append(faces_list_capturing, len(indexes))
            if not face_detected:
                if (len(boxes) > 0):
                    for i in range(len(boxes)):
                        if i in indexes:
                            x, y, w, h = boxes[i]
                            face_img = frame[y:y + h, x:x + w]
                            live = is_live_face(frame)
                            print('liveeee: \t', live)
                            if live:
                                count += 1
                                face_img = frame[y:y + h, x:x + w]
                                # Ensure the bounding box coordinates are within the frame dimensions
                                if x < 0: x = 0
                                if y < 0: y = 0
                                if x + w > frame.shape[1]: w = frame.shape[1] - x
                                if y + h > frame.shape[0]: h = frame.shape[0] - y

                                face_img = frame[y:y + h, x:x + w]

                                # Check if face_img is not empty
                                if face_img.size != 0:
                                    student_name = DoWork(face_img)

                                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                                    text_box = student_name.split("_")[0]
                                    # Input string
                                    input_string = text_box

                                    # Use regular expression to extract non-numeric part (byshel ay raqam fy el text bb2a 3ayza el esm bs)
                                    result = re.match("([a-zA-Z]+)", input_string)
                                    # If a match is found, result.group(1) retrieves the part of the string that matched the first capturing group in the pattern (the entire [a-zA-Z]+ part).
                                    if result:
                                        extracted_string = result.group(1)
                                    else:
                                        extracted_string = text_box
                                    '''el text aly byb2a mktob tht el rectangle aly fy el sora hyb2a student name bal el (_)
                                        3shan esm al shakhs byb2a abl el (_) w b3do byb2a el ID bta3 el shakhs'''
                                    cv2.putText(frame, extracted_string, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                                (255, 0, 0), 2)

                                    # Capture a photo when a face is detected
                                    photo_name = f'captured_photo_hh{count}.jpg'
                                    # el frame da bnhotlo esm el sora (photo_name)
                                    cv2.imwrite(photo_name, frame)
                                    if student_name == "unknown":
                                        pass
                                    else:
                                        # extract el student ID mn el filename w bhoto fy (number)
                                        number = int(student_name.split("_")[1].split(".")[0])
                                        if len(names_list_capturing) == 0:
                                            names_list_capturing = np.append(names_list_capturing, number)
                                            paths_list_capturing = np.append(paths_list_capturing, photo_name)
                                        else:
                                            if number in names_list_capturing:
                                                pass
                                            else:
                                                names_list_capturing = np.append(names_list_capturing, number)
                                                paths_list_capturing = np.append(paths_list_capturing, photo_name)
                                    photo_captured = True
        else:
            faces_list_capturing = np.append(faces_list_capturing, len(indexes))
            # No face detected
            face_detected = False
    # Display the result on the frame
    for i in range(len(boxes)):
        if i in indexes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Face Recognition for Attendance', frame)
    #time.sleep(0.01)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(names_list_capturing)
# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
