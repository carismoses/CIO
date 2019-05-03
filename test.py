import sys
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import tkinter
import shutil
import tempfile
import imageio
import scipy
import jupyter
import matplotlib

print('All dependencies met!')
