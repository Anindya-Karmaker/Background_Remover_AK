import sys
import os

# When the app is running from a PyInstaller bundle, sys._MEIPASS is the
# path to the temporary folder where the app's contents are extracted.
# We set the U2NET_HOME environment variable to point to the 'rembg_models'
# folder that we will bundle inside our app.
# This overrides the default rembg behavior of looking in the user's home directory.
if hasattr(sys, '_MEIPASS'):
    os.environ['U2NET_HOME'] = os.path.join(sys._MEIPASS, '_internal', 'rembg_models')