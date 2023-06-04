from streamlit_webrtc import webrtc_streamer
import streamlit as st
import matplotlib.pyplot as plt
import warnings
import os
from classify import Classifier
import threading
from PIL import Image
import datetime
import shutil

warnings.filterwarnings("ignore")

def log(classes,image_size,min_face_size,dropout_prob):
    with open('.tmp.txt','r') as mode:
        mode_=f'\t{mode.read()}'

    tree_={}
    for class_ in classes:
         tree_[class_]=len(os.listdir(f'Images/{class_}'))

    tree_=f'\t\t{tree_}'

    model_params_=f'\t\t\t[{image_size},{min_face_size},{dropout_prob}]'
    date_time_=f"\n{datetime.date.today()}\t{datetime.datetime.now().replace(microsecond=0).strftime('%H:%M:%S')}"

    data_=date_time_+mode_+model_params_+tree_

    header_='Date\t\t\tTime\t\tInput Mode\t\tParams\t\t\tClass Size'

    if 'logs.txt' not in os.listdir():
            data_=header_+data_

    with open('logs.txt','a') as f:
        f.write(data_)


def get_model():
    """
     Get model and plot it on the sidebar. This is a generator that can be used to iterate over the images in a streamed video.
    
     Returns: 
     	 A tuple of ( fig ax ) where fig is the figure and ax is the axis where the plot is
    """
    with open('.tmp.txt','r') as f:
        path_=f.read()
        if path_=='Web Cam':
            path_='train'

    if path_ in os.listdir('Images/'):
        classes=os.listdir(f'Images/{path_}')
        cols_=[]
        for col in classes:
            cols_.append('col_'+col)
        try:
            cols_=st.columns(len(classes))
        except:
            pass
    else:
        classes=[]
    global model
    # train_model if data. pt is not in os. listdir artifacts
    if "data.pt" not in os.listdir("Artifacts"):
        st.title("Train Model on Images")
        classes=os.listdir(f'Images/{path_}')
        if classes==[]:
            st.experimental_rerun()
        for class_ in classes:
            st.write(f'Class {class_}')
            image_row = []
            w_dir = f'Images/{path_}/{class_}/'
            imgs = os.listdir(w_dir)[:5]
            for img in imgs:
                image = Image.open(w_dir + img)
                resized_image = image.resize((100, 100))  # Resize the image to desired width and height
                image_row.append(resized_image)
            st.image(image_row, width=120)
        st.sidebar.title('Set Parameters')
        st.sidebar.text('Note: The `default` values\nrecommended.')
        #User parameters
        image_size=st.sidebar.select_slider('Image Size',range(128,264,16))
        min_face_size=st.sidebar.select_slider('Minimum Face Size',range(20,30))
        dropout_prob=st.sidebar.radio('Dropout Prob',[0.5,0.6,0.7,0.8,0.9])
        if st.sidebar.button('Train'):
            #progress bar
            progress_text = "Training... Please wait."
            bar=st.progress(0, text=progress_text)
            model=Classifier(image_size,min_face_size,dropout_prob)
            model.train_model()
            log(classes,image_size,min_face_size,dropout_prob)
            for i in range(10):
                bar.progress(i+1)

    lock = threading.Lock()
    img_container = {"img": None}

    def video_frame_callback(frame):
        """
        Callback for video frames. This is called every frame after it has been processed and can be used to save the image in the image container
        
        Args:
            frame: The frame that needs to be saved
        
        Returns: 
            The frame that has been saved to the image container for the next frame to be processed by the video
        """
        img = frame.to_ndarray(format="bgr24")
        with lock:
            img_container["img"] = img

        return frame

    if "data.pt" in os.listdir("Artifacts"):
        st.title("Class Classification.")
        choice_ = st.sidebar.radio(label="Choose", options=["Web Cam", "Upload"],key='test')

        if choice_=="Web Cam":
            ctx = webrtc_streamer(
                key="yo",
                video_frame_callback=video_frame_callback,
                sendback_audio=False,
            )

            fig_place = st.sidebar.empty()
            fig, ax = plt.subplots(1, 1)

            # Plot the face match of the image.
            while ctx.state.playing:
                with lock:
                    img = img_container["img"]
                # If img is None continue to do nothing.
                if img is None:
                    continue
                try:
                    res = model.face_match(img)
                except:
                    model=Classifier(160,20,0.6)
                    res = model.face_match(img)
                # If res is None continue.
                if res is None:
                    continue
                ax.cla()
                classes=os.listdir(f'Images/{path_}')
                w_=[1-res[1] for x in classes if x!=res[0]]
                w_.insert(classes.index(res[0]),res[1])   
                ax.barh(
                    y=classes,
                    width=w_,
                    color="red",
                )
                # ax.text(0.5, 0.25, f"{res[1]:.5f}", color="white", fontweight="bold")
                ax.set_title(f"Class - {res[0]} with distance = {res[1]:.5f})")
                fig_place.pyplot(fig)

        if choice_=='Upload':

            def re_upload():
                shutil.rmtree('Images/.garbage')
                os.mkdir('Images/.garbage')
                uploaded_files=st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"],accept_multiple_files=True)
                if uploaded_files:
                    for file in uploaded_files:
                        img_=file.getvalue()
                        with open(f"{'Images/.garbage/'}/{file.name}",'wb') as f:
                            f.write(img_)                
            
            if '.garbage' in os.listdir('Images/') and os.listdir('Images/.garbage') !=[]:
                model=Classifier(160,20,0.6)
                image_row = []
                preds=[]
                for img in os.listdir('Images/.garbage'):
                    image = Image.open(f'Images/.garbage/{img}')
                    resized_image = image.resize((100, 100))  # Resize the image to desired width and height
                    image_row.append(resized_image)
                    preds.append(model.face_match(image)[0])
                st.image(image_row, width=120,caption=preds)
                if st.button('Try Again.!'):
                    re_upload()
                
            else:
                re_upload()