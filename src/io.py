import numpy as np
from scipy.spatial import cKDTree

def read_xyz(filename):
	data = np.loadtxt(filename, skiprows=2, dtype=str)
	return data[:,1:].astype(float)


class PointCloud(object):
	def __init__(self,filename):	
		self.filename = filename

		if filename[-4:]=='.xyz':
			self.data = read_xyz(filename)

		
		self.coord =  self.data[:,:3]

	def unpack(self):
		return self.coord[:,0],self.coord[:,1],self.coord[:,2]



class Centres(PointCloud):
	def __init__(self,filename):
		super(Centres,self).__init__(filename)

		self.radius = self.data[:,3]
	
	def get_stress_info(self, stress_path,rcut=28):
		self.stress_path = stress_path
		self.major = np.loadtxt(stress_path+'major_stress.txt')
		self.minor = np.loadtxt(stress_path+'minor_stress.txt')
		self.trace = np.loadtxt(stress_path+'stress_trace.txt')
		self.aniso = major-minor
		self.pkl = pickle.load(open(stress_path+'stress_tensor.pkl', 'rb')) # 
		# filter only centres with more than one ontact
		self.tree = cKDTree(self.coord)
		self.neighs = tree.query_ball_tree(tree,rcut)
		self.coordination = np.array([len(n)-1 for n in neighs])
		# retain only the points with at least 2 neighbours
		assert len(self.coordination[self.coordination>1])==len(minor),"Jun's criterion failed"

		sel = coordination>1
		self.sel = sel
		self.stress_coord = self.coord[sel] 
		self.stress_coordination = self.coordination[sel]
		self.stress_tree = cKDTree(self.stress_coord)
		self.stress_neighs = tree.query_ball_tree(stress_tree,rcut)
		# local averages
		for avg,non  in zip([self.local_trace, self.local_maj, self.local_min,self.local_aniso], [self.major,self.minor,self.trace,self.aniso]) :

			avg = np.array([np.mean(self.non[n]) for n in self.stress_neighs])


class Contacts(PointCloud):
	def __init__(self,filename):
		super(Contacts,self).__init__(filename)

		self.volume = self.data[:,3]

		
