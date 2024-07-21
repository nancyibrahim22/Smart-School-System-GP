# array feha asamy kol el files
names_negative_list = []
# el array aly hyb2a feha el sewar kolha b3d el preprocessing
negative = []

# el sewar kolha b3d ma at3mlha encoding
negative1 = []
def PotentialYacta(video_file_path):
    import os
    import cv2
    import datetime
    output_directory = 'my_videos'
    IMAGE_HEIGHT, IMAGE_WIDTH = 224, 224
    SEQUENCE_LENGTH = 40
    import numpy as np
    import tensorflow as tf
    import cv2
    import numpy as np
    import os
    from PIL import Image
    from tensorflow.keras.applications.xception import preprocess_input
    import time
    import re
    encoder = tf.keras.models.load_model(r'VGG16encoder_lfw_casia_without_aug_delete1_max4_100epochs_early_printtrain.h5')

    def extractFrame():
        output_directory_vio = f'Violence_Incidences'
        os.makedirs(output_directory, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        output_filename_new = os.path.join(output_directory_vio, current_datetime)
        # video_file_path='t1'
        video_reader = cv2.VideoCapture(video_file_path)
        # print(video_reader)
        print(video_file_path)
        print("In Model")

        # Get the width and height of the video.
        original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # print(original_video_height)
        # print(original_video_width)
        # Declare a list to store video frames we will extract.
        frames_list = []

        # Store the predicted class in the video.
        predicted_class_name = ''

        # Get the number of frames in the video.
        video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        # print(video_frames_count)
        # Calculate the interval after which frames will be added to the list.
        skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)

        # Iterating the number of times equal to the fixed length of sequence.
        for frame_counter in range(SEQUENCE_LENGTH):

            # Set the current frame position of the video.
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

            success, frame = video_reader.read()
            # print(frame)
            if not success:
                print("break")
                break

            # Resize the Frame to fixed Dimensions.
            resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
            normalized_frame = resized_frame / 255.0
            # Appending the pre-processed frame into the frames list
            frames_list.append(normalized_frame)
        frames_list = np.array(frames_list)
        frames_list = np.expand_dims(frames_list, axis=0)  # Add batch dimension
        # print(frames_list)
        # Passing the  pre-processed frames to the model and get the predicted probabilities.
        # print(np.shape(np.expand_dims(frames_list, axis=0)))
        print("Frames list shape:", frames_list.shape)
        return frames_list

    def read_our_image(path):
        print(path)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(Image.fromarray(image).resize((128, 128)))

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default 1.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5)
        try:
            x, y, w, h = faces[0]
            face_img = image[y:y + h, x:x + w]
            face_img = np.array(Image.fromarray(face_img).resize((128, 128)))
        #         plt.imshow(face_img)
        #         plt.axis('off')  # Turn off axis
        #         plt.show()
        except:
            face_img = image
        #         print('********************************************')
        #         plt.imshow(face_img)
        #         plt.axis('off')  # Turn off axis
        #         plt.show()
        return face_img

    def read_anchor_path(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(Image.fromarray(image).resize((128, 128)))

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default 1.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5)
        try:
            x, y, w, h = faces[0]
            face_img = image[y:y + h, x:x + w]
            face_img = np.array(Image.fromarray(face_img).resize((128, 128)))
        #         plt.imshow(face_img)
        #         plt.axis('off')  # Turn off axis
        #         plt.show()
        except:
            face_img = image
        #         print('********************************************')
        #         plt.imshow(face_img)
        #         plt.axis('off')  # Turn off axis
        #         plt.show()
        return face_img



    # array
    neg_list_own_images = np.array([])



    '''el threshold aly hoa el distance ma ben el sora w ell anchor lw a2l mn 
    el threshold da or equals yb2a (0) positive w lw akbr yb2a (1) negative'''
    threshold = 1.5

    # el sum bta3 el distances bta3 kol sora m3 el anchor fy folder wahed
    neg_sum = 0

    def my_model():
        name = "Lesa Mt3mlsh"
        global names_negative_list
        global negative
        global negative1
        our_images_path = r'10 Shot'
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

        for i in range(len(negative)):
            negative1.append(encoder.predict(negative[i]))

    def DoWork(face_img):
        # array feha kol el distances bta3t el sewar m3 el anchor
        negative_distance_list = []
        global names_negative_list
        global neg_list_own_images
        global neg_sum
        distances = []
        final_names_avg = []
        global negative1
        anchor = []
        anchor.append(read_anchor_path(face_img))
        anchor = np.array(anchor)
        anchor = preprocess_input(anchor)
        tensor1 = encoder.predict(anchor)
        max_counter = 0

        # loop btgeb el distances kol sora m3 el anchor w nshof m3 el thershold hya positive wala negative
        for i in negative1:
            tensor3 = i
            neg_distance = np.sum(np.square(tensor1 - tensor3), axis=-1)
            negative_distance_list = np.append(negative_distance_list, neg_distance)
            neg_prediction = np.where(neg_distance <= threshold, 0, 1)
            neg_list_own_images = np.append(neg_list_own_images, neg_prediction)
        for i in range(len(negative_distance_list)):
            neg_sum += negative_distance_list[i]
            if (i + 1) % 5 == 0:
                neg_avg = neg_sum / 5
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
        if min_distance_avg > threshold:
            return "unknown"
        print(min_name)
        return min_name

    my_model()

    def run(try_anchor):
        my_model()
        anchor_img = cv2.imread(try_anchor)
        # anchor_img = cv2.cvtColor(anchor_img, cv2.COLOR_BGR2RGB)
        # student_name = DoWork(anchor_img)
        # Initialize Haar Cascade for face detection
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default 1.xml')

        if face_cascade.empty():
            print("Error: Could not load Haar Cascade Classifier.")
            exit()
        names_list_capturing = []
        gray = cv2.cvtColor(anchor_img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5)
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # Preprocess the detected face (resize, normalize, etc.) extracts the face region from the frame
                face_img = anchor_img[y:y + h, x:x + w]
                student_name = DoWork(face_img)
                cv2.rectangle(anchor_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
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
                cv2.putText(anchor_img, extracted_string, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (255, 0, 0), 2)
                # Capture a photo when a face is detected
                photo_name = f'D:\PycharmProjects\pythonProject21\captured_photo_anchor.jpg'
                # el frame da bnhotlo esm el sora (photo_name)
                cv2.imwrite(photo_name, anchor_img)
                if student_name == "unknown":
                    pass
                else:
                    # extract el student ID mn el filename w bhoto fy (number)
                    number = int(student_name.split("_")[1].split(".")[0])
                    if len(names_list_capturing) == 0:
                        names_list_capturing = np.append(names_list_capturing, number)
                    else:
                        if number in names_list_capturing:
                            pass
                        else:
                            names_list_capturing = np.append(names_list_capturing, number)
        print(names_list_capturing)
        print(photo_name)
        return photo_name, names_list_capturing

    frames=extractFrame()
    # print(f'frameees list: {frames} \n')
    # print(f'frameees list len: {len(frames)} \n')

    frames_img = []
    names_frames = []
    count=0
    for frame_set in frames:  # Iterate through sets of 40 frames
        for frame in frame_set:  # Iterate through each frame in the set
            print(f'frameees shape: {frame.shape} \n')
            count+=1
            photo_name = f'captured_photo_frame_{count}.jpg'
            # el frame da bnhotlo esm el sora (photo_name)
            cv2.imwrite(photo_name, (frame * 255).astype(np.uint8))
            img, names = run(frame)
            frames_img = np.append(frames_img,img)
            names_frames = np.append(names_frames,names)

    # try_anchor = r"D:\PycharmProjects\pythonProject21\9 shot\Nancy\Nancy6.jpg"
    # run(try_anchor)


PotentialYacta(f'Violence_Incidences/viotest2.mp4')