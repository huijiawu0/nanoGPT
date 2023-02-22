import os
import numpy as np
from tqdm import tqdm


# Load the list of file names
import sys
folder = sys.argv[1]
file_list = [os.path.join(folder, n) for n in os.listdir(folder)]
# Merge all the files
arr_len = 0
for f in tqdm(file_list):
    arr = np.memmap(f, dtype=np.uint16, mode='r')
    arr_len += len(arr)
    print(arr, len(arr))

print(arr_len)

total_arr = np.memmap('train.bin', dtype=np.uint16, mode='w+', shape=(arr_len,))
idx = 0
for f in tqdm(file_list):
    a1 = np.memmap(f, dtype=np.uint16, mode='r')
    total_arr[idx: idx + len(a1)] = a1
    idx += len(a1)
    
