from streamlit_webrtc import webrtc_streamer
import streamlit as st
import matplotlib.pyplot as plt
import os
from classify import Classifier, build_model, build_ml_model, Classifier_ml
import threading
from PIL import Image
import datetime
import shutil
import torch
import cv2
import gc


def log(
    classes,
    num_epochs=None,
    lr=None,
    accuracy=None,
    test_acc=None,
    average_loss=None,
    test_loss=None,
    ml_=None,
):
    """
    Logs information about the performance of the model. This is a function to be called from the console and not directly from the command line

    @param classes - A list of class names
    @param num_epochs - The number of epochs to run the model
    @param lr - The learning rate to use for the model.
    @param accuracy - The accuracy of the model. It is a float between 0 and 1
    @param test_acc - The accuracy of the model. It is a float between 0
    @param average_loss
    @param test_loss
    """
    with open(".tmp.txt", "r") as f:
        path_ = f.read()
        # path_ Web Cam train path_ train
        if path_ == "Web Cam":
            path_ = "train"

    with open(".tmp.txt", "r") as mode:
        mode_ = f"\t{mode.read()}"

    tree_ = {}
    # Returns the number of images in the classes.
    for class_ in classes:
        tree_[class_] = len(os.listdir(f"Images/{path_}/{class_}"))

    tree_ = f"\t\t{tree_}"

    date_time_ = f"\n{datetime.date.today()}\t{datetime.datetime.now().replace(microsecond=0).strftime('%H:%M:%S')}"

    if ml_ == None:
        perf_ = f"{accuracy}, {test_acc}, {average_loss}, {test_loss}"
        model_params_ = f"\t\t\t[{num_epochs},{lr}]\t\t"
        data_ = date_time_ + mode_ + model_params_ + tree_ + "\t\t" + perf_
        header_ = "Date\t\t\tTime\t\tInput Mode\t\tEpochs,LR\t\t\tClass Size\t\tModel's Perfomance"
        # If logs. txt is not in os. listdir os. listdir logs. txt
        if "logs.txt" not in os.listdir():
            data_ = header_ + data_
        with open("logs/logs.txt", "a") as f:
            f.write(data_)

    if ml_ != None:
        perf_ml_ = f"{ml_}"
        data_ = date_time_ + mode_ + tree_ + "\t\t\t" + perf_ml_
        header_ = "Date\t\t\tTime\t\tInput Mode\t\tClass Size\t\tModel's Perfomance"
        if "logs2.txt" not in os.listdir():
            data_ = header_ + data_
        with open("logs/logs2.txt", "a") as f:
            f.write(data_)


# ------------------------------------Inferencing---------------------------------------------
def inference():
    # ------------------------------Using Web Cam-----------------
    lock = threading.Lock()
    img_container = {"img": None}

    def video_frame_callback(frame):
        """
        Callback for video frames. This is called every frame after it has been processed and can be used to save the image in the image container
        @param frame - The frame that needs to be saved
        @return The frame that has been saved to the image container for the next frame to be processed by the video
        """
        img = frame.to_ndarray(format="bgr24")
        with lock:
            img_container["img"] = img

        return frame

    # Plot the classification of the image.
    st.title("Class Classification.")
    choice_ = st.sidebar.radio(
        label="Choose", options=["Web Cam", "Upload"], key="inference"
    )

    # Plot the face match of the image.
    if choice_ == "Web Cam":
        torch.cuda.empty_cache()
        gc.collect()

        ctx = webrtc_streamer(
            key="Inferencing",
            video_frame_callback=video_frame_callback,
            sendback_audio=False,
        )

        log1 = open("logs.txt", "r").read().split("\t")
        dl_acc = log1[-1]
        log2 = open("logs2.txt", "r").read().split("\t")
        ml_acc = log2[-1]
        choice_ml_dl = st.radio(
            label="Choose",
            options=[
                f"Using DL (epochs-train-test-acc-loss: {dl_acc}",
                f"Using ML(Acc - {ml_acc})",
            ],
            key="Using Cam",
        )

        fig_place = st.sidebar.empty()
        fig, ax = plt.subplots(1, 1)

        # Plot the face match of the image.
        # Draw classes in the background.
        while ctx.state.playing:
            with lock:
                img = img_container["img"]
            # If img is None continue to do nothing.
            # Draw a classifier image.
            if img is not None:
                torch.cuda.empty_cache()  # Clear's gpu cache

                if choice_ml_dl == f"Using DL (epochs-train-test-acc-loss: {dl_acc}":
                    res = Classifier(img)
                    ax.cla()
                    classes = os.listdir(f"Images/{path_}")
                    # If res is None continue.
                    # Plot the class and its accuracy.
                    if res != None:
                        ax.cla()
                        classes = os.listdir(f"Images/{path_}")
                        w_ = [1 - res[1][0] for x in classes if x != classes[res[0][0]]]
                        w_.insert(classes.index(classes[res[0][0]]), res[1][0])
                        ax.barh(y=classes, width=w_[0], color="red", align="edge")
                        ax.set_title(
                            f"Class - {classes[res[0][0]]} with Accuracy = {max(res[1][0]):.5f})"
                        )
                        fig_place.pyplot(fig)

                if choice_ml_dl == f"Using ML(Acc - {ml_acc})":
                    res = Classifier_ml(img)
                    if all(res) != None:
                        ax.cla()
                        classes = os.listdir(f"Images/{path_}")
                        w_ = [1 - res[1] for x in classes if x != classes[res.argmax()]]
                        print(classes[res.argmax()], res[1])
                        w_.insert(res.argmax(), res[1])
                        ax.barh(y=classes, width=w_[0], color="red", align="edge")
                        ax.set_title(
                            f"Class - {classes[res.argmax()]} with Accuracy = {max(res):.5f})"
                        )
                        fig_place.pyplot(fig)

    # ---------------------Using Upload--------------------------------

    # This function is used to create a cache of images and images.
    if choice_ == "Upload":
        torch.cuda.empty_cache()
        gc.collect()

        def rm_():
            """
            Remove files and directories created by st. cache_data and st. cache_files.
            This is a destructive operation
            """
            st.cache_data.clear()
            try:
                shutil.rmtree("Images/.garbage")
            except:
                pass
            try:
                os.mkdir("Images/.garbage")
            except:
                pass

        def re_upload():
            """
            Upload images to new location and delete old one if there is any.
            This is useful for testing.
            """
            rm_()
            uploaded_files = st.file_uploader(
                "Upload Image",
                type=["jpg", "png", "jpeg"],
                accept_multiple_files=True,
                key="reupload",
            )
            # Remove all files from the uploaded_files.
            if uploaded_files:
                rm_()
                # Write all the images in the uploaded_files to the garbage directory.
                for file in uploaded_files:
                    img_ = file.getvalue()
                    with open(f"{'Images/.garbage/'}/{file.name}", "wb") as f:
                        f.write(img_)

        uploaded_files = st.file_uploader(
            "Can choose the images from `test.zip`.",
            type=["jpg", "png", "jpeg"],
            accept_multiple_files=True,
            key="test_upload",
        )
        # This function will resize the images to desired width and height
        if uploaded_files:
            rm_()
            # Write all the images in the uploaded_files to the garbage directory.
            for file in uploaded_files:
                img_ = file.getvalue()
                with open(f"{'Images/.garbage/'}/{file.name}", "wb") as f:
                    f.write(img_)

            # Create a new image and add it to the test suite.
            if (
                ".garbage" in os.listdir("Images/")
                and os.listdir("Images/.garbage") != []
            ):
                image_row = []
                preds = []
                classes = os.listdir(f"Images/{path_}")
                # Resize the images to desired width and height.
                for img in os.listdir("Images/.garbage"):
                    image = cv2.imread(f"Images/.garbage/{img}")
                    image_ = Image.open(f"Images/.garbage/{img}")
                    resized_image = image_.resize(
                        (100, 100)
                    )  # Resize the image to desired width and height
                    image_row.append(resized_image)
                    torch.cuda.empty_cache()
                    gc.collect()

                    log1 = open("logs.txt", "r").read().split("\t")
                    dl_acc = log1[-1]
                    log2 = open("logs2.txt", "r").read().split("\t")
                    ml_acc = log2[-1]
                    choice_ml_dl = st.radio(
                        label="Choose",
                        options=[
                            f"Using DL (epochs-train-test-acc-loss: {dl_acc}",
                            f"Using ML(Acc - {ml_acc})",
                        ],
                        key="Using upload",
                    )
                    if (
                        choice_ml_dl
                        == f"Using DL (epochs-train-test-acc-loss: {dl_acc}"
                    ):
                        res = Classifier(image)
                        acc = f"({max(res[1][0]):.5f})"
                        preds.append(f"{classes[int(res[0])]}{acc}")
                    if choice_ml_dl == f"Using ML(Acc - {ml_acc})":
                        res = Classifier_ml(image)
                        acc = f"({max(res):.5f})"
                        preds.append(f"{classes[res.argmax()]}{acc}")
                # if image_row is empty or not an image row
                if image_row == []:
                    st.experimental_rerun()
                st.image(image_row, width=120, caption=preds)

            else:
                re_upload()

        # Download test images and download test images.
        if "test" in os.listdir("Images/"):
            shutil.make_archive("Images/test", "zip", "Images/test")
            file_path = "Images/test.zip"
            with open(file_path, "rb") as file:
                file_contents = file.read()
            st.download_button(
                "Download Test Images", data=file_contents, file_name="test_imgs.zip"
            )


# ----------------------------------------------------- Training model----------------------------------


def get_model():
    """
    Get the model to train on. This is a function for use in the script that will be called by the : py :
    func : ` ~gensim. models. get_model ` function.
    @return A tuple of ( model_path class_names ) where model_path is the path to the model file
    """
    global path_, model

    with open(".tmp.txt", "r") as f:
        path_ = f.read()
        # path_ Web Cam train path_ train
        if path_ == "Web Cam":
            path_ = "train"

    # Return a list of columns for each image in the images directory.
    if path_ in os.listdir("Images/"):
        classes = os.listdir(f"Images/{path_}")
        cols_ = []
        # Add the class names to the list of classes
        for col in classes:
            cols_.append("col_" + col)
        try:
            cols_ = st.columns(len(classes))
        except:
            pass
    else:
        classes = []

    if "data.pt" or "data.pkl" not in os.listdir("Artifacts"):
        # -----------------------Using DL-------------------------------------------
        # train_model if data. pt is not in os. listdir artifacts
        # Train Model on Classes.
        if "data.pt" not in os.listdir("Artifacts"):
            st.title("Train Model on Classes")
            tree_tr = {}
            tree_te = {}
            classes = os.listdir(f"Images/{path_}")
            for class_ in classes:
                tree_tr[class_] = len(os.listdir(f"Images/{path_}/{class_}"))
            for class_ in classes:
                tree_te[class_] = len(os.listdir(f"Images/test/{class_}"))
            st.text(f"Train data : {tree_tr}, Test data : {tree_te}")
            st.header("Set Parameters")
            st.text(
                "Note: The higher `Epochs` value, Consumes more time and resources,\nbut will have +ve impact on models performance."
            )
            # User parameters
            num_epochs = st.select_slider("Epochs", range(10, 100, 10))
            lr = st.select_slider("Learning Rate", [0.0001, 0.001, 0.01, 0.1, 1])
            st.write(
                """ML - Using SVC
                    with C=100, class_weight=None, gamma="auto", kernel="rbf", probability=True.
                    \nDL - Using VGG19 Transfer learning."""
            )
            # Train the model and log the training data.
            if st.button("Train", key="dl"):
                gc.collect()
                ml_acc = build_ml_model()
                if ml_acc != None:
                    print(ml_acc)
                    log(classes=classes, ml_=ml_acc)
                torch.cuda.empty_cache()
                accuracy, test_acc, average_loss, test_loss = build_model(
                    num_epochs, lr
                )  # Train
                log(
                    classes,
                    num_epochs,
                    lr,
                    accuracy,
                    test_acc,
                    average_loss,
                    test_loss,
                )  # log fn
