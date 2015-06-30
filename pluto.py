"""
Methods to read and visualize PLUTO's output.
"""


import pyPLUTO as pp
import mayavi.mlab as mlab
import pylab, numpy
import nemmen




def movie(fname="movie.avi"):
	import fish
	import os, fnmatch
	import subprocess

	# count the number of snapshots to create the movie
	nfiles=0
	for file in os.listdir('.'):
		if fnmatch.fnmatch(file, 'plot*.jpeg'):
			nfiles=nfiles+1	

	# creates ascii list of files
	#cmd="ls plot.*.jpeg | sort -n -t . -k 2 > list.txt"
	#subprocess.call(cmd.split())

	# Progress bar initialization
	peixe = fish.ProgressFish(total=nfiles)
	
	# snapshot creation
	for i in range(0,nfiles-1):
		#cutplane(i)
		volume(i)
		peixe.animate(amount=i)








class Pluto:
	"""
	Class that defines data objects imported from PLUTO.
	
	The object's attributes are:
	
	- x1,x2,x3
	- v1,v2,v3
	- pressure p
	- rho
	- n1,n2,n3

	
	To read the simulation output for frame 10:
	
	>>> import pluto
	>>> p=pluto.Pluto(10)
	
	Plots density field:
	
	>>> p.snap()
	"""
		
	def __init__(self, i=0):
		d=pp.pload(i)
		
		# when getting xi below, assumes uniform grid
		if d.n1>1: 
			self.x1=numpy.linspace(d.dx1[0],d.dx1[-1],d.n1)
			self.v1,self.n1=d.vx1,d.n1
			self.speed=numpy.sqrt(self.v1*self.v1)
		if d.n2>1: 
			self.x2=numpy.linspace(d.dx2[0],d.dx2[-1],d.n2)
			self.v2,self.n2=d.vx2,d.n2
			self.speed=numpy.sqrt(self.v1*self.v1 + self.v2*self.v2)
		if d.n3>1: 
			self.x3=numpy.linspace(d.dx3[0],d.dx3[-1],d.n3)
			self.v3,self.n3=d.vx3,d.n3
			self.speed=numpy.sqrt(self.v1*self.v1 + self.v2*self.v2 + self.v3*self.v3)

		self.p=d.prs
		self.rho=d.rho
		self.pp=d # pypluto object
		self.frame=i
		# this is probably incorrect
		self.Mdot=-4.*numpy.pi*self.x1**2*self.rho*self.v1

	

	def snap(self,var=None):
		"""
Creates snapshot of 2D simulation generated in cartesian coordinates.
If 'var' is not specified, plots density field.

>>> p=pluto.Pluto(10)
>>> p.snap(10,p.p)
		"""
		import seaborn
		seaborn.set_style({"axes.grid": False})
		cmap=seaborn.cubehelix_palette(light=1, as_cmap=True)

		lw = 5*self.speed/self.speed.max()
    
		#x=range(0,self.n1)
		#X,Y=numpy.meshgrid(x,x)
    
		pylab.clf()
		# transposes the array because imshow is weird
		if var==None:
			pylab.imshow(self.rho.T, cmap=cmap, extent=[self.x1[0],self.x1[-1],self.x2[0],self.x2[-1]])
		else:
			pylab.imshow(var.T, cmap=cmap, extent=[self.x1[0],self.x1[-1],self.x2[0],self.x2[-1]])	
		pylab.colorbar()
		#streamplot(X,Y,v2,v1,color='k',linewidth=lw)
		#streamplot(X,Y,v2,v1,color='k')
		pylab.savefig('plot.'+str(self.frame)+'.jpeg')


	def pol2cart(self, n=200):
		"""
Creates a new object with variables in cartesian coordinates.
Useful if the object was created by PLUTO in polar coordinates.

n is the new number of elements n^2.
		"""
		# creates copy of current object which will have the new
		# coordinates
		obj=Pluto(self.frame)
	
		# r, theta
		r,th=self.x1,self.x2

		xnew=numpy.linspace(-r.max(), r.max(), n)
		ynew=xnew.copy()	
		rho=numpy.zeros((n,n))	
		p=rho.copy()
		
		# goes through new array
		# I am sure this can severely sped up
		for i in range(xnew.size):
		    for j in range(ynew.size):
        		rnew,thnew=nemmen.cart2pol(xnew[i],ynew[j])
        		# position in old array
        		iref=nemmen.search(rnew, r)
        		jref=nemmen.search(thnew, th)
        		rho[i,j]=self.rho[iref,jref]		
        		p[i,j]=self.p[iref,jref]	
	
		obj.x1,obj.x2=xnew,ynew
		obj.rho,obj.p=rho,p
		
		return obj	



#### 3d routines


def cutplane(i):
	"""
Snapshot of 3d cartesian simulation, generating cut planes.	

i : index corresponding to frame you want to plot
	"""
	d=pp.pload(i)
	x1,x2,x3=d.x1,d.x2,d.x3
	v1,v2,v3=d.vx1,d.vx2,d.vx3
	p=d.prs
	rho=d.rho

	mlab.clf()
	#mlab.figure(size=(600,600))

	# volume rendering
	mp=mlab.pipeline.scalar_field(p)
	mrho=mlab.pipeline.scalar_field(rho)
	#mlab.pipeline.volume(mp)#,vmax=rho.max()/5.)

	# streamlines
	#flow = mlab.flow(v1, v2, v3, seed_scale=0.5, seed_resolution=8, integration_direction='both',seed_visible=False)

	# cut planes
	mlab.pipeline.image_plane_widget(mp, plane_orientation='y_axes', slice_index=100)
	mlab.pipeline.image_plane_widget(mp, plane_orientation='x_axes', slice_index=100)

	# move camera to appropriate distance
	dcam=mlab.view()[2] # distance of camera to center
	##mlab.move(forward=dcam/2.)
	mlab.view(distance=dcam/2.)

	# saves snapshot
	mlab.savefig('plot.'+str(i)+'.jpeg',size=(800,800))




def volume(i):
	"""
Volume rendering of 3d cartesian simulation.	
	
i : index corresponding to frame you want to plot
	"""
	d=pp.pload(i)
	x1,x2,x3=d.x1,d.x2,d.x3
	v1,v2,v3=d.vx1,d.vx2,d.vx3
	p=d.prs
	rho=d.rho

	mlab.clf()

	# volume rendering
	mp=mlab.pipeline.scalar_field(p)
	mrho=mlab.pipeline.scalar_field(rho)
	mlab.pipeline.volume(mp)#,vmax=rho.max()/5.)

	# streamlines
	#flow = mlab.flow(v1, v2, v3, seed_scale=0.5, seed_resolution=8, integration_direction='both',seed_visible=False)

	# cut planes
	#mlab.pipeline.image_plane_widget(mp, plane_orientation='y_axes', slice_index=100)
	#mlab.pipeline.image_plane_widget(mp, plane_orientation='x_axes', slice_index=100)

	# move camera to appropriate distance
	dcam=mlab.view()[2] # distance of camera to center
	##mlab.move(forward=dcam/2.)
	mlab.view(distance=dcam/2.)

	# saves snapshot
	mlab.savefig('plot.'+str(i)+'.jpeg',size=(800,800))




