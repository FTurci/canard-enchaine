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
		self.tree = cKDTree(self.coord)

	def get_gr(self,dr=1.0,rmax=100, factor=3):
		r = np.arange(dr,rmax,dr)
		c=self.coord
		self.tree = cKDTree(c)
		nb = self.tree.count_neighbors(self.tree,r, cumulative=False)
		# print(nb[0])
		nb[0] = 0 # ignore self-distances
		# V = np.prod(self.coord.ptp(axis=0))
		N  = c.shape[0]

		# ideal gas
		
		accumulate = np.zeros_like(nb,dtype=float)
		
		ideal = np.random.uniform(self.coord.min(axis=0), self.coord.max(axis=0)*factor,size=(N*factor**3,3))
		ideal_tree = cKDTree(ideal)
		accumulate = np.array(ideal_tree.count_neighbors(ideal_tree,r, cumulative=False))/factor**3
		# accumulate-= accumulate[0]
		# print(accumulate,repeat)
		# accumulate[0]=1
		# g = nb/(4*np.pi*r**2*dr)*V/(N*(N-1))
		g = nb/accumulate 
		gr={}
		gr['r'] = r
		gr['g'] = g
		gr['dr'] = dr
		gr['rmax'] = rmax
		gr['ideal'] = accumulate
		gr['nb'] = nb
		# plt.plot(r,nb)
		# plt.plot(r,accumulate, '.')
		# gr['V'] = V
		self.gr = gr
	
	def get_stress_info(self, stress_path,rcut=28):
		self.stress_path = stress_path
		self.major = np.loadtxt(stress_path+'major_stress.txt')
		self.minor = np.loadtxt(stress_path+'minor_stress.txt')
		self.trace = np.loadtxt(stress_path+'stress_trace.txt')
		self.validids = np.loadtxt(stress_path+'particle_stress_index.txt').astype(int)
		self.aniso = self.major-self.minor
		self.pkl = pickle.load(open(stress_path+'stress_tensor.pkl', 'rb')) # 
		# filter only centres with more than one ontact
		
		self.neighs = self.tree.query_ball_tree(self.tree,rcut)
		self.coordination = np.array([len(n)-1 for n in self.neighs])
		# retain only the points with at least 2 neighbours
		# assert len(self.coordination[self.coordination>1])==len(self.minor),"Jun's criterion failed"

		sel = self.validids #self.coordination>1
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

	def plot_sample_slices(self,z,quantity1,quantity2,thickness=30, feature_scaling=True,border=None):

		xyz = self.stress_coord
		# trick to get the names right
		coordination = self.stress_coordination
		local_trace = self.local_trace 
		neg_local_trace = -self.local_trace 
		local_major = self.local_major 
		neg_local_major = -self.local_major 
		local_minor = self.local_minor 
		neg_local_minor = -self.local_minor 
		
		local_aniso = self.local_aniso 
		trace = self.trace
		neg_trace = -self.trace
		major = self.major
		neg_major = -self.major
		minor = self.minor
		neg_minor = -self.minor
		aniso = self.aniso


		select = (xyz[:,-1]>z-thickness*0.5)*(xyz[:,-1]<z+thickness*0.5)
		XYZ = xyz[select]
		x = XYZ[:,0]
		y = XYZ[:,1]

		q1 = quantity1[select]
		q2 = quantity2[select]
		# print(locals().keys())
		q1_name = [ k for k,v in locals().items() if np.all(v == quantity1)][0]
		q2_name = [ k for k,v in locals().items() if np.all(v == quantity2)][0]
		if feature_scaling :
			q1 = (q1-q1.mean())/q1.std()
			q2 = (q2-q2.mean())/q2.std()
			vmin = -2
			vmax = 2
		else:
			vmin = vmax= None

		fig,ax = plt.subplots(1,3, figsize=(12,4))

		s0 = ax[0].scatter(x,y,c=q1, vmin=vmin,vmax=vmax,edgecolor='k')
		s1 = ax[1].scatter(x,y,c=q2, vmin=vmin,vmax=vmax,edgecolor='k')
		ax[0].set_title(q1_name)
		ax[1].set_title(q2_name)
		ax[0].axis('equal')
		ax[1].axis('equal')
		fig.colorbar(s0,ax=ax[0])
		fig.colorbar(s1,ax=ax[1])


		if border==None:
			select_q1 = q1
			select_q2 = q2
		else:
			valid = (x>x.min()+border)*(x<x.max()-border)*(y>y.min()+border)*(y<y.max()-border)
			select_q1 = q1[valid]
			select_q2 = q2[valid]
		ax[2].hist2d(select_q1,select_q2)
		ax[2].set_xlabel(q1_name)
		ax[2].set_ylabel(q2_name)
		pearsonr = scipy.stats.pearsonr(select_q1,select_q2)[0]
		ax[2].set_title(f'Pearson r={pearsonr:.2f}')
		
		fig.suptitle(f'Slice z={z:.1f} of thickness {thickness}',y=1.05)


	def get_masked_gr(self,quantity,threshold):
		if hasattr(self,'gr')==False:
			self.get_gr()

		r = self.gr['r']
		dr = self.gr['dr']
		coord = self.stress_coord
		valid = quantity<=threshold

		mask_tree = cKDTree(coord[valid])
		mask_nb = mask_tree.count_neighbors(mask_tree,r, cumulative=False)#<=threshold)
		N  = len(quantity[valid])

		assert N==coord[valid].shape[0],f"Mismatching N: {N} {coord[valid].shape}"

		accumulate = np.zeros_like(mask_nb,dtype=float)

		# for k in range(repeat):
		if N>5000:
			factor = 1
			ideal = np.random.uniform(self.stress_coord[valid].min(axis=0), self.stress_coord[valid].max(axis=0),size=(N,3))
		else:
			factor = 3
			ideal = np.random.uniform(self.stress_coord[valid].min(axis=0), self.stress_coord[valid].max(axis=0)*factor,size=(N*factor**3,3))
		ideal_tree = cKDTree(ideal)
		accumulate = np.array(ideal_tree.count_neighbors(ideal_tree,r, cumulative=False))/factor**3
		
		# accumulate /= repeat
		# g = nb/(4*np.pi*r**2*dr)*V/(N*(N-1))
		masked_g = mask_nb/accumulate 
		

		# masked_g = mask_nb/(4*np.pi*r**2*dr)*V/(N*(N-1))
		masked_g[0] = 0
		masked_gr = {}
		masked_gr['r'] = r
		masked_gr['g'] = masked_g
		masked_gr['dr'] = dr
		return masked_gr

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

		
