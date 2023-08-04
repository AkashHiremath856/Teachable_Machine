import streamlit as st
import os
import shutil
import random
from streamlit_webrtc import webrtc_streamer
import av
from datetime import datetime
from PIL import Image
import cv2
import numpy as np


# ------------------ Preprocessing ------------------#
class preprocessing:
    def __init__(self):
        self.img_dir = "Images/train/"
        self.classes = os.listdir(self.img_dir)
        class_names = os.listdir(self.img_dir)
        # Returns a list of all class names in the training image.
        if not class_names:
            class_names = os.listdir(self.img_dir)
        self.class_names = class_names
        self.test_dir = "Images/test/"
        self.face_cascade_path = (
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.face_cascade = cv2.CascadeClassifier(self.face_cascade_path)

    # Class balance
    def class_balance(self):
        class_size = {}
        # Add the size of each class in the class_names list.
        for class_name in self.class_names:
            class_size[class_name] = len(os.listdir(self.img_dir + class_name))
        print(class_size)
        max_diff = max(class_size.values())
        max_class = []
        [max_class.append(k) for k, v in class_size.items() if v == max_diff]
        for class_name in self.class_names:
            if class_name != max_class[0]:
                w_dir = self.img_dir + class_name + "/"
                diff = max_diff - len(os.listdir(w_dir))
                choices = random.choices(os.listdir(w_dir), k=diff)
                for choice in choices:
                    name_ = choice.split(".")[0]
                    new_name = choice.replace(name_, name_ + "_copy")
                    shutil.copy(
                        w_dir + choice,
                        w_dir + new_name,
                    )
        self.roi()

    # Region of intrest
    def roi(self):
        for class_ in os.listdir(self.img_dir):
            for img_ in os.listdir(f"{self.img_dir}/{class_}"):
                img_array = cv2.imread(f"{self.img_dir}/{class_}/{img_}")
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    img_array = img_array[y : y + h, x : x + w]
                    cv2.imwrite(f"{self.img_dir}/{class_}/{img_}", img_array)
                    if img_array.shape[0] < 100:
                        os.remove(f"{self.img_dir}/{class_}/{img_}")

        if len(os.listdir(self.img_dir + os.listdir(self.img_dir)[0])) < 12:
            self.augment_()
        else:
            self.split_images()

    def split_images(self):
        # Splitting Images into train,test
        try:
            os.mkdir(self.test_dir)
        except:
            pass
        # Move images from the test directory to the test directory.
        for class_ in self.class_names:
            _images = os.listdir(self.img_dir + class_)
            total_images = len(_images)
            choice = random.choices(_images, k=int(total_images * 0.3))  # 30% test
            if class_ not in os.listdir(self.test_dir):
                os.mkdir(self.test_dir + "/" + class_)
            # Move all files in choice to the test directory
            for i in choice:
                try:
                    shutil.move(
                        self.img_dir + class_ + "/" + i,
                        self.test_dir + "/" + class_,
                    )
                except:
                    continue

    def augment_image(self, image, output_dir, prefix):
        resized = image

        # Rotate
        angle = np.random.randint(-15, 15)
        rows, cols, _ = resized.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated = cv2.warpAffine(resized, M, (cols, rows))

        # Flip
        flip_direction = np.random.randint(
            0, 3
        )  # 0 = horizontal flip, 1 = vertical flip, 2 = both flips
        flipped = cv2.flip(rotated, flip_direction - 1)

        # Save augmented image
        output_path = os.path.join(output_dir, f"{prefix}_augmented.jpg")
        cv2.imwrite(output_path, flipped)
        # print(f"Augmented image saved: {output_path}")

    def augment_images_in_directory(self, input_dir, output_dir):
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Loop through images in input directory
        for filename in os.listdir(input_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(input_dir, filename)

                # Load image
                image = cv2.imread(image_path)

                # Augment image
                self.augment_image(image, output_dir, os.path.splitext(filename)[0])

    def augment_(self):
        # Specify input and output directories
        w_dir = self.img_dir

        for class_ in os.listdir(w_dir):
            input_directory = f"{w_dir}/{class_}"
            output_directory = f"{w_dir}/{class_}"
            self.augment_images_in_directory(input_directory, output_directory)
        self.split_images()


# ------------------ Capture Images ------------------#


def cap(cname):
    """
    Capture and store images for a class. This is a streaming function that can be used to capture and store images for a class.

    Args:
        cname: Name of the class. If empty the title will be printed to standard output.

    Returns:
        A tuple of ( VideoFrame stream ) where video frame is a video stream and stream is a VideoFrame
    """
    # This function will create a video frame for the given class.
    image_row = []
    train_dir = "Images/train/"
    if cname != "":
        st.title(f"For Class {cname}")
        w_dir = train_dir + cname
        # Create a directory for the training directory if it doesn t exist.
        if "train" in os.listdir("Images") and cname not in os.listdir(train_dir):
            os.makedirs(w_dir)

        def video_frame_callback(frame):
            """
            Callback for VideoFrame. This is called every frame in the video. We write the frame to disk and return a VideoFrame that can be used to train the model

            Args:
                frame: The frame to be saved

            Returns:
                The video frame that was saved to disk and used to train the model ( if needed ). Note that the frame is saved in bgr
            """
            img = frame.to_ndarray(format="bgr24")
            tim = datetime.now().time().second
            nam = f"{train_dir}/{cname}/frame_{str(tim)}.jpg"
            cv2.imwrite(nam, img)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(
            key=f"{cname}_1",
            video_frame_callback=video_frame_callback,
            sendback_audio=False,
        )
        # Create a row of images to be used in the post - processing wizard. This is a copy of the code that was copied from django. contrib. image. service

        image_row = []
        for img in os.listdir(f"{train_dir}/{cname}"):
            try:
                image = Image.open(f"{train_dir}/{cname}/{img}")
                # Resize the image to desired width and height
                resized_image = image.resize((100, 100))
                image_row.append(resized_image)
            except:
                continue
        if (
            len(os.listdir(f"{train_dir}/{cname}")) < 8
            and len(os.listdir(f"{train_dir}/{cname}")) >= 1
        ):
            st.warning("Add more images to improve models performance. ")
        nu_ = list(zip(image_row, range(1, len(image_row) + 1)))
        st.image(image_row, width=120, caption=[x[1] for x in nu_])

        if os.listdir(f"{train_dir}/{cname}"):
            if st.button("Clear", key=cname + "i"):
                st.warning("Deleted Images.")
                shutil.rmtree(f"{train_dir}/{cname}")
