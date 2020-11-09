import argparse
from pathlib import Path
import os
import numpy as np
from scipy.spatial import cKDTree

path = Path(__file__).parent.parent

parse = argparse.ArgumentParser(description='Merge XYZ coord file and stress information')
parse.add_argument('polymer', type=str)
parse.add_argument('rcut', type=float)
args = parse.parse_args()

pol = args.polymer
rcut = args.rcut

datapath = path/'data'/f'phi_p_{pol}'

stresspath = path/'data'/'stressTensor'/f'{pol}'

assert os.path.exists(datapath), f'{datapath} does not exist'
assert os.path.exists(stresspath), f'{datapath} does not exist'


coords = np.loadtxt(str(datapath)+'/coords.xyz', usecols=[1,2,3], skiprows=2)
major = np.loadtxt(str(stresspath)+'/major_stress.txt')
minor = np.loadtxt(str(stresspath)+'/minor_stress.txt')
trace = np.loadtxt(str(stresspath)+'/stress_trace.txt')

#  guess Jun's condition for particles to be retained

#guessing the criterion used by Jun
print("cutoff =",rcut)
centre_tree = cKDTree(coords)
neighs = centre_tree.query_ball_tree(centre_tree,rcut)
coordination = np.array([len(n)-1 for n in neighs])

assert len(coordination[coordination>1]) == len(minor), "Mismatching  no. of stresses and no. of particles. Change cutoff?"

coords = coords[coordination>1]


with open(str(stresspath)+'/coordstress.xyz', 'w') as fw:
    fw.write('%d\nParticles\n'%coords.shape[0])
    for i,p in enumerate(coords):
        fw.write(f'A {p[0]} {p[1]} {p[2]} {major[i]} {minor[i]} {trace[i]}\n')