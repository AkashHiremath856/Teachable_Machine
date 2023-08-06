# importing libraries
import streamlit as st
import os
from streamlit import session_state as state
from capture import preprocessing, cap
from classify_ui import get_model, inference
from upload_ import upload, display_info, css_
import shutil
import base64
from PIL import Image
import datetime
from bing_image_downloader import downloader


title = "Teachable Machine"
st.set_page_config(
    page_title=title,
    page_icon="Artifacts/pytorch.png",
    initial_sidebar_state="auto",
)


def reset(ky="default", i=0):
    """
    Remove files and directories if opt is None or button " Reset " is pressed. This is useful for debugging the program

    @param opt - Option to reset or
    """
    if st.button("Reset", key=ky):
        st.warning("Warning: The session will be restarted!.")
        st.cache_data.clear()

        try:
            os.remove("Artifacts/data.pt")
        except:
            pass
        try:
            os.remove("Artifacts/data.pkl")
        except:
            pass
        try:
            shutil.rmtree("Images/test")
        except:
            pass
        try:
            os.remove("Images/test.zip")
        except:
            pass
        try:
            shutil.rmtree("Images/train")
        except:
            pass
        try:
            os.remove("Images/test.zip")
        except:
            pass

        if i == 0:
            state.page = "home"


# ------------------ Class home_page ------------------#
class home_page:
    def __init__(self):
        self.session_timeout = range(-2, 3)
        self.img_dir = "Images/"
        os.makedirs(self.img_dir, exist_ok=True)

    # --------choice----------------
    def choices(self):
        choice = st.sidebar.radio(
            label="Choose",
            options=["Web Cam", "Upload", "Web Scrap"],
            key="input_mode" + str(0),
        )
        return choice

    # ------------------Home Page content------------------------
    def info(self, n_classes):
        if n_classes < 2:
            display_info(n_classes)  # About
            # Convert the image bytes to a data URL
            image_bytes = open("Artifacts/git.png", "rb").read()
            encoded = base64.b64encode(image_bytes).decode()
            data_url = f"data:image/png;base64,{encoded}"
            st.write(
                f'<a title="Source Code" href="https://github.com/AkashHiremath856/Teachable_Machine" target="blank"><img src="{data_url}" class="icon" alt="Github"></a>',
                unsafe_allow_html=True,
            )
            st.write(
                f"<footer class='footer'>© 2023 Teachable Machine™. All rights reserved.</footer>",
                unsafe_allow_html=True,
            )

    # ------------------ Data input UI --------------------------#
    def _ui(self, choice):
        st.title(title)
        self.w_dir = self.img_dir + "train/"
        self.n_classes = st.sidebar.select_slider(
            "Number of classes", range(1, 10), key=choice
        )
        self.info(self.n_classes)

        # Assign a class name for each class
        if self.n_classes > 1:
            self.classes = []
            # Create the training images if they don't exist. This is called after the session is created
            if self.w_dir not in os.listdir(self.img_dir):
                os.makedirs(self.w_dir, exist_ok=True)
            if os.listdir(self.w_dir) != []:
                cls_ = os.listdir(self.w_dir)[0]
                if os.listdir(f"{self.w_dir}/{cls_}") != []:
                    l_file = os.listdir(f"{self.w_dir}/{cls_}")[0]
                    m_time = os.path.getmtime(f"{self.w_dir}/{cls_}/{l_file}")
                    m_minute = datetime.datetime.fromtimestamp(m_time).minute
                    diff_ = datetime.datetime.now().minute - m_minute
                    if diff_ not in self.session_timeout:
                        st.warning("Please End Previous Session first.")
                        reset(44)

            # Add a class name to the list of classes.
            for _ in range(self.n_classes):
                txt = "Assign a class name for class " + str(_ + 1)
                class_name = st.sidebar.text_input(txt)
                # Add a class name to the list of classes.
                if class_name is not None:
                    self.classes.append(class_name)

            # ------------------Web Cam------------
            if "" not in self.classes and choice == "Web Cam":
                # Cap the classes in the list of classes.
                for names in self.classes:
                    cap(names)

            # ------------------Upload------------------
            if "" not in self.classes and choice == "Upload":
                global k
                k = 0
                # Upload all classes to the server.
                for names in self.classes:
                    k += 1
                    upload(names, str(k))

            # -----------------Web scrap------------
            if choice == "Web Scrap" and "" not in self.classes:
                ds_size = st.select_slider(
                    "Number of Images per class (Web Scrapping)", range(0, 100, 10)
                )
                if ds_size >= 10:
                    if st.button("Web Scrap"):
                        st.info("Please wait while Web Scrapping.", icon="ℹ️")
                        for class_ in self.classes:
                            downloader.download(
                                class_,
                                limit=ds_size,
                                output_dir=self.w_dir,
                                verbose=False,
                            )
                        for names in self.classes:
                            image_row = []
                            os.makedirs
                            imgs = os.listdir(self.w_dir + names)
                            # resize the images to 100 pixels
                            for img in imgs:
                                image = Image.open(self.w_dir + names + "/" + img)
                                resized_image = image.resize((100, 100))
                                image_row.append(resized_image)
                            nu_ = list(zip(image_row, range(1, len(image_row) + 1)))
                            st.title(f"Class {names}")
                            st.image(image_row, width=120, caption=[x[1] for x in nu_])

            # Preprocessing and Train Button
            # This function is called when the sidebar button is pressed.
            if (
                sorted(self.classes) == sorted(os.listdir(self.w_dir))
                and os.listdir(f"{self.w_dir}/{self.classes[-1]}") != []
            ):
                if "test" not in os.listdir(self.img_dir):
                    # This function is called when the button is clicked.
                    if st.sidebar.button("Next", key="Next-btn"):
                        obj2 = preprocessing()
                        obj2.class_balance()
                        st.toast("Preprocessing...Done!", icon="ℹ️")
                        if "test" in os.listdir("Images"):
                            state.page = "train"

    def train_page(self):
        get_model()
        # Download the model. pt button.
        if "data.pkl" and "data.pt" in os.listdir("Artifacts/"):
            file_path = "Artifacts/data.pt"
            inference()
            with open(file_path, "rb") as file:
                file_contents = file.read()
            st.download_button("Export Model", data=file_contents, file_name="model.pt")
            # Reset the page to home if the button Reset is pressed
        reset()


# This function is called by the main loop to build the model.
if __name__ == "__main__":
    st.markdown(css_(), unsafe_allow_html=True)  # load css
    obj = home_page()
    # Set the page to home (Navigation).
    if "page" not in state:
        state.page = "home"
    # This function is called by the user when the page is home or train.
    # home page
    if state.page == "home":
        home_page()
        choice = obj.choices()
        obj._ui(choice)

    # train page
    elif state.page == "train":
        obj.train_page()
