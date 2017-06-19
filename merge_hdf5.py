import h5py
import numpy as np

file1 = '/scratch_net/bmicdl03/code/python/ACDC_challenge_refactored/newdata_288x288.hdf5'
file2 = '/scratch_net/bmicdl03/code/python/ACDC_challenge_refactored/newwenjiadata_288x288.hdf5'

out_file = '/scratch_net/bmicdl03/code/python/ACDC_challenge_refactored/newwenjia_acdc_merged.hdf5'

data1 = h5py.File(file1, 'r')
data2 = h5py.File(file2, 'r')

h5py_file_out = h5py.File(out_file, "w")

# [print(key) for key in data2.keys()]

for key in data1.keys():

    if key in data2.keys():

        datum1 = data1[key][:]
        datum2 = data2[key][:]

        # if not datum1.shape[0] == 0 and not datum2.shape[0] == 0:

        print(key)

        print(datum1.shape)
        print(datum2.shape)

        if datum1.shape[0] > 0 and datum2.shape[0] == 0:
            h5py_file_out.create_dataset(key, data=datum1)
        elif datum1.shape[0] == 0 and datum2.shape[0] > 0:
            h5py_file_out.create_dataset(key, data=datum2)
        else:
            h5py_file_out.create_dataset(key, data=np.concatenate([datum1, datum2], axis=0))

h5py_file_out.close()