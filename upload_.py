from pathlib import Path
import streamlit as st
import os
import shutil
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = (
    True  # PIL is reading in blocks of the file and that it expects
)
# that the blocks are going to be of a certain size.
# It turns out that you can ask PIL to be tolerant of files that are truncated
# (missing some file from the block) by changing a setting.


def upload(cname, k):
    """
    Upload images to class'cname'and return a list of images. This is a wrapper around st. file_uploader and does not check for errors

    @param cname - name of class to upload images to
    @param k - number of images to upload to class e. g
    """
    w_dir = "Images/train/"
    if "train" not in os.listdir("Images"):
        os.makedirs(w_dir, exist_ok=True)
    path = Path(w_dir + cname)
    path.mkdir(exist_ok=True)
    if os.listdir(w_dir + cname) == []:
        st.title(f"Class {cname}")
        uploaded_files = st.file_uploader(
            "Upload Image",
            type=["jpg", "png", "jpeg"],
            accept_multiple_files=True,
            key=str(k) + cname,
        )
        # Writes the uploaded images to the file.
        if uploaded_files:
            # Write the images to the file.
            for file in uploaded_files:
                img_ = file.getvalue()
                with open(f"{path}/{file.name}", "wb") as f:
                    f.write(img_)

    image_row = []
    w_dir_ = w_dir + cname + "/"
    imgs = os.listdir(w_dir_)
    # resize the images to 100 pixels
    for img in imgs:
        image = Image.open(w_dir_ + img)
        resized_image = image.resize((100, 100))
        image_row.append(resized_image)

    # Add more images to improve models performance.
    if len(os.listdir(w_dir + cname)) < 8 and len(os.listdir(w_dir + cname)) >= 1:
        st.warning("Add more images to improve models performance. ")
    nu_ = list(zip(image_row, range(1, len(image_row) + 1)))
    st.image(image_row, width=120, caption=[x[1] for x in nu_])

    # Delete all images in the cache.
    if os.listdir(w_dir + cname):
        # Clear the cache data.
        if st.button("Clear", key=cname + "i"):
            st.warning("Deleted Images.")
            shutil.rmtree(w_dir + cname)
            st.cache_data.clear()


# About Teachable Machine
def display_info(n_classes):
    """
    Displays information about Teachable Machine. This is a function to be called from the GUI and should not be called externally

    @param n_classes - number of classes to
    """
    # if n_classes 1 print out the text
    if n_classes <= 1:
        st.text(
            """
            With Teachable Machine, you can train a model to recognize and classify different 
            inputs, such as images. The tool uses a technique called transfer learning, where a 
            pre-trained model is adapted to recognize new classes of data.
            
            Here's a general overview of how Teachable Machine works:

                1. Collection: You collect and label examples of different classes of data. 
                For example, if you're training an image classifier, you would provide images for 
                each category you want to recognize.

                2. Training: Teachable Machine uses your labeled examples to train a machine 
                learning model. It leverages a pre-existing neural network and fine-tunes it 
                using your data. The training is done in your web browser and doesn't require 
                any server-side processing.

                3. Testing: Once the model is trained, you can test it using new examples to see 
                how well it performs. Teachable Machine provides a live preview that shows the 
                model's predictions in real-time.

                4. Exporting: You can export the trained model in different formats, to use your 
                model.

            Teachable Machine can be used for various applications, such as creating custom 
            image classifiers, gesture recognition systems, or sound classifiers. It's a 
            beginner-friendly tool that provides a hands-on introduction to machine learning 
            concepts.
                """
        )


# ---------------css style-----------------------#
def css_():
    style = """
        <style>
        #MainMenu{visibility: hidden;}
        .css-uf99v8,.css-1avcm0n{background-color:black;}
        .css-cio0dv,.css-cio0dv a{color:black}
        .css-10trblm{font-size: xx-large;top:-25px}
        *{font-family: 'Comfortaa'}
        .css-183lzff {font-family: 'Comfortaa'}
        .footer {position: fixed; left: 0; bottom: 0;width: 100%;text-align: center;padding: 20px;}
       .icon{position: absolute;top: -800px;right: 50px;height:64px;width:64px}
        .icon::after {content: attr(title);display: none;position: absolute; top: 100%;left: 50%;transform: translateX(-50%);
        padding: 5px 10px;background-color: black;color: white;font-size: 14px;white-space: nowrap;}
        .icon:hover::after {cursor: pointer; display: block; }
        .css-183lzff{font-size:16px}
        </style>
        """
    return style
