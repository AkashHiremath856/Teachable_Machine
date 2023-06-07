import streamlit as st
import os
import shutil
import random
from streamlit_webrtc import webrtc_streamer
import av
from datetime import datetime
from PIL import Image


# ------------------ Preprocessing ------------------#
class preprocessing:
    def __init__(self, img_dir, class_names=None):
        """
         Initialize the class with training and testing data. This is called by __init__ and should not be called directly
         
         Args:
         	 img_dir: Directory where images are stored
         	 class_names: List of class names to be used
        """
        self.classes = os.listdir(img_dir)
        self.img_dir = img_dir
        # Returns a list of all class names in the training image.
        if not class_names:
            class_names = os.listdir(self.img_dir)
        self.class_names = class_names
        self.test_dir='Images/test'

    # Class balance
    def class_balance(self):
        """
         Balance the classes by copying frames to different class files. This is done in a recursive function.
        """
        class_size = []
        # Add the size of each class in the class_names list.
        for class_name in self.class_names:
            w_dir_ = self.img_dir + class_name + "/"
            class_size.append(len(os.listdir(w_dir_)))
        max_index = class_size.index(max(class_size))
        # Copy all classes in the training directory to the train directory.
        for class_name in self.class_names:
            w_dir_ = self.img_dir + class_name + "/"
            size_ = len(os.listdir(w_dir_))
            differ = class_size[max_index] - size_
            # Copy random choices from the current class size to the current class size.
            if size_ != class_size[max_index]:
                choices = random.choices(os.listdir(w_dir_), k=differ)
                # Copy all the choices in the choices list to the current directory.
                for choice in choices:
                    shutil.copy(
                        w_dir_ + choice,
                        w_dir_ + choice.replace("frame", "frame_copy"),
                    )
            size_ = len(os.listdir(w_dir_))
            # If the size of the class is different than the max_index
            if size_ != class_size[max_index]:
                self.class_balance()

    def split_images(self):
        """
         Split images into train and test. This is done by picking a random number of images from the set of images to be
        """
        # Splitting Images into train,test
        try:
            os.mkdir(self.test_dir)
        except:
            pass
        total_images = len(os.listdir(self.img_dir + self.class_names[0]))
        # Move images from the test directory to the test directory.
        if os.listdir(self.test_dir):
            # Print out the number of files in the image directory.
            for root, dirs, files in os.walk(self.img_dir + "/"):
                print(f"{root}/{len(files)}")
        else:
            choice = random.choices(
                range(total_images), k=int(total_images * 0.2)
            )  # 20% test
            # Move all classes in the class_names to the test_dir
            for class_ in self.class_names:
                files = os.listdir(self.img_dir + class_)
                # Move all files in choice to the test directory
                for i in choice:
                    try:
                        shutil.move(
                            self.img_dir + class_ + "/" + files[i],
                            self.test_dir,
                        )
                    except:
                        continue
            # st.write("The data split is as follows :")
            # for root, dirs, files in os.walk(self.img_dir):
            #     st.write(f"Class {root} has {len(files)} Images")
            # st.write(f"Class Images/test/ has {len(os.listdir('Images/test/'))}")

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
    image_row=[]
    if cname != "":
        st.title(f"For Class {cname}")
        w_dir = "Images/train/" + cname
        # Create a directory for the training directory if it doesn t exist.
        if cname not in os.listdir("Images/train/"):
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
            nam = f"Images/train/{cname}/frame_{str(tim)}.jpg"
            im = Image.fromarray(img.astype('uint8'), 'RGB')
            im.save(nam)
            return av.VideoFrame.from_ndarray(img, format="bgr24") #------------------

        webrtc_streamer(
            key=f"{cname}_1",
            video_frame_callback=video_frame_callback,
            sendback_audio=False,
        )
        # Create a row of images to be used in the post - processing wizard. This is a copy of the code that was copied from django. contrib. image. service
        
        image_row = []
        for img in os.listdir(f'Images/train/{cname}') :
            try:
                image = Image.open(f'Images/train/{cname}/{img}')
                resized_image = image.resize((100, 100))  # Resize the image to desired width and height
                image_row.append(resized_image)
            except:
                continue
        if len(os.listdir(f'Images/train/{cname}'))<8 and len(os.listdir(f'Images/train/{cname}'))>=1:
            st.warning('Add more images to improve models performance. ')
        nu_=(list(zip(image_row,range(1,len(image_row)+1))))
        st.image(image_row, width=120,caption=[x[1] for x in nu_])

        if os.listdir(f'Images/train/{cname}'):
            if st.button('Clear',key=cname+'i'):
                st.warning('Deleted Images.')
                shutil.rmtree(f'Images/train/{cname}')
