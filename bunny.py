import numpy as np
import urllib.request
import tempfile
import pywavefront

bunny = None

def get_bunny():
    global bunny
    if bunny is None:
        url = 'https://graphics.stanford.edu/~mdfisher/Data/Meshes/bunny.obj'
        response = urllib.request.urlopen(url)
        data = response.read()
        text = data.decode('utf-8')
        with tempfile.NamedTemporaryFile(mode='w', dir='/tmp', delete=False) as tmpfile:
            tmpfile.write(text)
            bunny = np.array(tuple(pywavefront.Wavefront(tmpfile.name).vertices),
                             dtype=np.float32)
    return bunny
