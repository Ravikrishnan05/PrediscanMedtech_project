# PrediscanMedtech_project

PrediscanMedtech_project

🔥 Problem:



1) Error that i have got 

EOFError                                  Traceback (most recent call last)
<ipython-input-7-5b104c4af0e7> in <cell line: 0>()
      1 from tensorflow.keras.models import load_model
      2 
----> 3 model = load_model('/content/full_retina_model.h5', compile=False)

10 frames
/usr/local/lib/python3.11/dist-packages/keras/src/utils/python_utils.py in func_load(code, defaults, closure, globs)
     81     except (UnicodeEncodeError, binascii.Error):
     82         raw_code = code.encode("raw_unicode_escape")
---> 83     code = marshal.loads(raw_code)
     84     if globs is None:
     85         globs = globals()

EOFError: EOF read where object expected

2)
model = tf.keras.models.load_model('/content/full_retina_model.h5')
But that only works for full models saved with model.save(...).

Your file was likely saved using:
model.save_weights('full_retina_model.h5')

3)I gave this 
import h5py

with h5py.File('/content/full_retina_model.h5', 'r') as f:
    print("Keys in HDF5 file:")
    print(list(f.keys()))

and got 
Keys in HDF5 file:
['model_weights', 'optimizer_weights']
