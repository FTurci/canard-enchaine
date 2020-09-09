import numpy as np

def read_xyz(filename):
	data = np.loadtxt(filename, skiprows=2, dtype=str)
	return data[:,1:].astype(float)


class PointCloud:
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
		

class Contacts(PointCloud):
	def __init__(self,filename):
		super(Contacts,self).__init__(filename)

		self.volume = self.data[:,3]

