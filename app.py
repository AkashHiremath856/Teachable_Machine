# importing libraries
import streamlit as st
import os
from streamlit import session_state as state
from capture import preprocessing, cap
from classify_ui import get_model
from upload_ import upload,display_info
import shutil
import base64
from pathlib import Path
import datetime


title='Teachable Machine'

st.set_page_config(page_title=title, page_icon='Artifacts/pytorch.png',initial_sidebar_state='auto')

session_timeout=[-2,-1,0,1,2]

img_dir=Path('Images')
img_dir.mkdir(exist_ok=True)


def reset(i=0):
    """
     Remove files and directories if opt is None or button " Reset " is pressed. This is useful for debugging the program
     
     @param opt - Option to reset or
    """
    if st.button("Reset"):
                st.warning('Warning: The session will be restarted!.')
                st.cache_data.clear()

                try:
                    os.remove('Artifacts/data.pt') 
                except:
                    pass
                try:
                    shutil.rmtree('Images/test') and os.mkdir('Images/test') 
                except:
                    pass
                try:
                    os.remove('Images/test.zip') 
                except:
                    pass
                try:
                    shutil.rmtree('Images/train') and os.mkdir('Images/train')
                except:
                    pass
                try:
                    shutil.rmtree('Images/upload') and os.mkdir('Images/upload')
                except:
                    pass
                try:
                    shutil.rmtree('Images/.garbage') and os.mkdir('Images/.garbage')
                except:
                    pass
                try:
                    os.remove('Images/test.zip') 
                except:
                    pass
                
                if i!=0:
                    pass
                else:
                    state.page='home'
                    home_page(i)

def info(n_classes):
    if n_classes==1:
        display_info(n_classes) #About
        st.write(f'<a title="Source Code" href="https://github.com/AkashHiremath856/Teachable_Machine" target="blank"><img src="{data_url}" class="icon" alt="Github"></a>', unsafe_allow_html=True)
        st.write(f"<footer class='footer'>© 2023 Teachable Machine™. All rights reserved.</footer>",unsafe_allow_html=True)

# ------------------ Web_cam ------------------#
def home_page(i):
    """
     This function is called when the user clicks on the home page. It creates the folder and selects 
     the number of classes. Streamlit is a dialog for choosing the number of classes to assign a 
     class name to the first 10 classes and a list of class names for each class.
     
     @return The url to the folder where the images are stored on Silverpeas. If there is no image_dir 
     it returns
    """
    global image_dir
    global n_classes

    choice = st.sidebar.radio(label="Choose", options=["Web Cam", "Upload"],key='train'+str(i))
    image_dir=None
    # This function is called when the user clicks on the main page.
    # This function is called by the main loop to build the model.
    if choice == "Web Cam":
        st.title(title)
        # ------------------ Streamlit ------------------#
        n_classes = st.sidebar.select_slider("Number of classes", range(1, 10))
        info(n_classes)

        # Assign a class name for each class
        if n_classes > 1:
            classes = []
            # Create the training images if they don't exist. This is called after the session is created
            if 'train' not in os.listdir('Images'):
                os.mkdir('Images/train')
            if os.listdir(f'Images/train/')!=[]:
                cls_=os.listdir(f'Images/train/')[0]
                if os.listdir(f'Images/train/{cls_}')!=[]:
                    l_file=os.listdir(f'Images/train/{cls_}')[0]
                    m_time=os.path.getmtime(f'Images/train/{cls_}/{l_file}')
                    m_minute=datetime.datetime.fromtimestamp(m_time).minute
                    diff_=datetime.datetime.now().minute-m_minute
                    if diff_ not in session_timeout:
                        st.warning('Please End Previous Session first.')
                        reset(44)

            # Add a class name to the list of classes.
            for _ in range(n_classes):
                txt = "Assign a class name for class " + str(_ + 1)
                class_name = st.sidebar.text_input(txt)
                # Add a class name to the list of classes.
                if class_name is not None:
                    classes.append(class_name)             
            # Cap the classes in the list of classes.
            for names in classes:
                    cap(names)
            
            if classes !=[]:
                # if the sidebar button train button is pressed and the sidebar button train button is pressed.
                if (
                    classes == os.listdir("Images/train")
                    and os.listdir(f"Images/train/{classes[-1]}") != []
                ):
                    st.sidebar.text("Note: Click all the\nbuttons twice.")
                    # This function is called when the button is clicked.
                    if st.sidebar.button("Next", key=names):
                        img_dir = "Images/train/"
                        class_names_ = os.listdir(img_dir)
                        obj2 = preprocessing(img_dir, class_names_)
                        obj2.class_balance()
                        obj2.split_images()
                        state.page = "train"
                    reset()
            

    # This function is called when the user clicks on the upload button.
    if choice == "Upload":
        st.title(title)

        n_classes_ = st.sidebar.select_slider("Number of classes", range(1, 10),key='upload')
        info(n_classes_)

        # This function is called by the main loop to generate classes.
        if n_classes_ > 1:
            classes_ = []
            k=0
            # Assign a class name for each class.
            for _ in range(n_classes_):
                txt = "Assign a class name for class " + str(_ + 1)
                class_name_ = st.sidebar.text_input(txt)
                # Add the class name to the list of classes.
                if class_name_ is not None:
                    classes_.append(class_name_)
            # This function is called by the main loop.
            if 'upload' not in os.listdir('Images'):
                os.mkdir('Images/upload')
            if os.listdir(f'Images/upload/')!=[]:
                cls_=os.listdir(f'Images/upload/')[0]
                if os.listdir(f'Images/upload/{cls_}')!=[]:
                    l_file=os.listdir(f'Images/upload/{cls_}')[0]
                    m_time=os.path.getmtime(f'Images/upload/{cls_}/{l_file}')
                    m_minute=datetime.datetime.fromtimestamp(m_time).minute
                    diff_u=datetime.datetime.now().minute-m_minute
                    if diff_u not in session_timeout:
                        reset(78)
                        st.warning('Please End Previous Session first.')

            if '' not in classes_:
                # Upload all classes to the server.
                for names in classes_:
                        k+=1
                        upload(names,str(k))
                # This function is called when the sidebar button is pressed.
                try:
                    if (
                    classes_ == os.listdir("Images/upload")
                    and os.listdir(f"Images/upload/{classes_[-1]}") != []
                    ):
                        st.sidebar.text("Note: Click all the\nbuttons twice.")
                        # This function is called when the button is clicked.
                        if st.sidebar.button("Next", key=names):
                            img_dir = "Images/upload/"
                            class_names_ = os.listdir(img_dir)
                            obj2 = preprocessing(img_dir, class_names_)
                            obj2.class_balance()
                            obj2.split_images()
                            state.page = "train"
                except:
                    try:
                        os.mkdir("Images/upload")
                    except:
                        pass
    return choice


def train_page():
    """
     Trains the model and sets the page to the home page if reset button is pressed. This is a hack to work around the problem that the user cannot enter a reset
    """
    get_model()
    # Download the model. pt button.
    if 'data.pt' in os.listdir('Artifacts/'):
        file_path='Artifacts/data.pt'
        with open(file_path, 'rb') as file:
            file_contents = file.read()
        st.download_button('Export Model',data=file_contents,file_name='model.pt')
    # Reset the page to home if the button Reset is pressed
    reset()

def main():
    """
     Main function of the program. If there is no page in the state it will default to home. Otherwise it will call the appropriate function
    """
    i=0
    # Set the page to home.
    if "page" not in state:
        state.page = "home"

    # This function is called by the user when the page is home or train.
    if state.page == "home":
        i+=1
        tmp=home_page(i)
        
        # Write the tmp file to the. tmp. txt file.
        if tmp is not None:
            with open('.tmp.txt','w') as f:
                f.write(tmp)

    elif state.page == "train":
        train_page()

#---------------css style-----------------------#
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
st.markdown(style, unsafe_allow_html=True)

git_path='Artifacts/git.png'

def read_image_file(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return data

# Convert the image bytes to a data URL
def convert_image_to_data_url(image_bytes):
    encoded = base64.b64encode(image_bytes).decode()
    data_url = f"data:image/png;base64,{encoded}"
    return data_url

# Read the image file and convert it to a data URL
image_bytes = read_image_file(git_path)
data_url = convert_image_to_data_url(image_bytes)

# main function for the main module
if __name__ == "__main__":
    main()
