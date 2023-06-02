import os
import shutil

try:
    os.remove('Artifacts/data.pt')
except:
    pass
try:
    shutil.rmtree('Images/train')
except:
    pass
try:
    shutil.rmtree('Images/upload')
except:
    pass

os.system('streamlit run main_ui.py')