import numpy as np
import sys
data = np.loadtxt(sys.argv[1])

with (open(sys.argv[1]+'.xyz', 'w')) as fw:
    fw.write('%d\nAtoms\n'%data.shape[0])
    for line in data:
        fw.write("A ")
        for word in line:
            fw.write(f"{word} ")
        fw.write('\n')
