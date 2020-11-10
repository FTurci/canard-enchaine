import numpy as np
from scipy.spatial import cKDTree
import pickle
import ipyvolume as ipv
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import scipy

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
		self.aniso = self.major-self.minor
		self.pkl = pickle.load(open(stress_path+'stress_tensor.pkl', 'rb')) # 
		# filter only centres with more than one ontact
		self.tree = cKDTree(self.coord)
		self.neighs = self.tree.query_ball_tree(self.tree,rcut)
		self.coordination = np.array([len(n)-1 for n in self.neighs])
		# retain only the points with at least 2 neighbours
		assert len(self.coordination[self.coordination>1])==len(self.minor),"Jun's criterion failed"

		sel = self.coordination>1
		self.sel = sel
		self.stress_coord = self.coord[sel] 
		self.stress_coordination = self.coordination[sel]
		self.stress_tree = cKDTree(self.stress_coord)
		self.stress_neighs = self.stress_tree.query_ball_tree(self.stress_tree,rcut)
		# local averages

		self.local_trace = np.array([np.mean(self.trace[n]) for n in self.stress_neighs])
		 
		self.local_major = np.array([np.mean(self.major[n]) for n in self.stress_neighs])
		self.local_minor = np.array([np.mean(self.minor[n]) for n in self.stress_neighs])
		self.local_aniso = np.array([np.mean(self.aniso[n]) for n in self.stress_neighs])

	def plot_clusters(self,threshold, rcut=28, size=5.):
		ipv.clear()

		s = self.trace<=threshold
		xyz = self.stress_coord[s]

		x, y, z = xyz[:,0],xyz[:,1],xyz[:,2]
		clustering = DBSCAN(eps=rcut, min_samples=1).fit(xyz)
		labels = np.unique(clustering.labels_)

		counts,edges= np.histogram(clustering.labels_,bins=np.arange(labels.min(), labels.max()+1))

		largest = edges[counts.argmax()]

		color_dict = {}
		for l in labels:
			color_dict[l] =  tuple(np.random.uniform(0,1, size=3))
			
		color_dict[largest] = (1,1,1)
		colors = np.array([color_dict[c] for c in clustering.labels_])

		color = np.array([(x-x.min())/x.ptp(),np.ones_like(x), np.ones_like(x)]).T
		ipv.scatter(x, y, z, size=size, marker="sphere", color=colors)

		ipv.show()

	def weighted_gr(self,threshold):
		plt.clf()
		fig,ax= plt.subplots(1,2);
		dr =0.25
		r = np.arange(dr,80,dr)
		coord = self.stress_coord
		valid = self.trace<=threshold
		wcentre_tree = cKDTree(coord[valid])
		weight = self.trace[self.trace<=threshold]
		wnb = wcentre_tree.count_neighbors(wcentre_tree,r, cumulative=False, weights=weight)#<=threshold)
		V = np.prod(coord.ptp(axis=0))
		N  = len(self.trace[valid])
		assert N==coord[valid].shape[0],f"Mismatching N: {N} {coord[valid].shape}"
		gw = wnb/(4*np.pi*r**2*dr)*V/(N*(N-1))
		gw[0] = 0
		ax[0].plot(r,gw/gw[-1]); 
		ax[1].hist(self.trace[valid],bins=32);
		ax[0].set_xlabel('r [pixels]'); 
		ax[0].set_ylabel(r'$g(r)$');
		ax[1].set_yscale('log');

	def pearsonr_coordination_z_slice(self,z,quantity, thickness=30, markersize=80, plot=False, ax=None,row=None):

		xyz = self.stress_coord
		coordination = self.stress_coordination
		select = (xyz[:,-1]>z)*(xyz[:,-1]<z+thickness)
		XYZ = xyz[select]
		pearsonr = scipy.stats.pearsonr(quantity[select],coordination[select])[0]


		if plot:
			ax[row,0].scatter(XYZ[:,0],XYZ[:,1],c=-(trace[select]-trace[select].mean())/trace[select].std(),s=markersize,vmin=-2,vmax=2);
			ax[row,1].scatter(XYZ[:,0],XYZ[:,1],c=(coordination[select]-coordination[select].mean())/coordination[select].std(),s=markersize,vmin=-2,vmax=2);
			ax[row,0].set_title(f'z={z} Scaled trace')
			ax[row,1].set_title(f'z={z} Coordination, pearsonr={pearsonr}')
		# print(z, pearsonr)
		return pearsonr


	def pearson_profile(self,zs,quantity,thickness=30):
		ps = []
		for i,z in enumerate(zs):
			ps.append(self.pearsonr_coordination_z_slice(z,quantity,thickness))
		return ps

	def all_pearson_profiles(self,zs):
		results={}
		results['z'] = zs
		results['trace_profile'] = self.pearson_profile(zs,self.local_trace)
		results['aniso_profile'] = self.pearson_profile(zs,self.local_aniso)
		results['major_profile'] = self.pearson_profile(zs,self.local_major)
		results['minor_profile'] = self.pearson_profile(zs,self.local_minor)
		return results



class Contacts(PointCloud):
	def __init__(self,filename):
		super(Contacts,self).__init__(filename)

		self.volume = self.data[:,3]

		
