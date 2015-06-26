"""
Methods to read and visualize PLUTO's output.
"""


import pyPLUTO as pp
import mayavi.mlab as mlab
import pylab, numpy




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
	
	# put here to command to create the snapshots
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
		
		self.x1,self.x2,self.x3=d.x1,d.x2,d.x3
		self.dx1,self.dx2,self.dx3=d.dx1,d.dx2,d.
		self.v1,self.v2,self.v3=d.vx1,d.vx2,d.vx3
		self.p=d.prs
		self.rho=d.rho
		self.n1,self.n2,self.n3=d.n1,d.n2,d.n3

		self.speed = numpy.sqrt(self.v1*self.v1 + self.v2*self.v2 + )
		lw = 5*speed/speed.max()	
	


def polar2cartesian(r, t, grid, x, y, order=3):
	"""
http://stackoverflow.com/questions/2164570/reprojecting-polar-to-cartesian-grid
	"""
	from scipy.interpolate import interp1d
	from scipy.ndimage import map_coordinates
	
	return map_coordinates(grid, numpy.array([r, t]),
		order=order).reshape(r.shape)





def snapPolar(i):
	"""
Creates snapshot of 2D simulation generated in polar coordinates.

>>> snapPolar(10)
	"""
	import seaborn
	seaborn.set_style({"axes.grid": False})
	cmap=seaborn.cubehelix_palette(light=1, as_cmap=True)

	#x=range(0,nx)
	#X,Y=numpy.meshgrid(x,x)
	r,t=x1,x2
	x, y = pol2cart(r,t)   
	im=polar2cartesian(r, t, rho, x, y, order=0)

	pylab.clf()
	pylab.xlim(0,n1)
	pylab.ylim(0,n2)
	pylab.imshow(im, cmap=cmap)
	pylab.colorbar()
	#streamplot(X,Y,v2,v1,color='k',linewidth=lw)
	#streamplot(X,Y,v2,v1,color='k')
	pylab.savefig('plot.'+str(i)+'.jpeg')







def snap(i):
    """
Creates snapshot of 2D simulation generated in cartesian coordinates.

>>> snapPolar(10)
    """
    import seaborn
    seaborn.set_style({"axes.grid": False})
    cmap=seaborn.cubehelix_palette(light=1, as_cmap=True)

    d=pp.pload(i)
    x1,x2=d.x1,d.x2
    dx1,dx2=d.dx1,d.dx2
    v1,v2=d.vx1,d.vx2
    p=d.prs
    rho=d.rho
    nx=d.n1
    
    speed = numpy.sqrt(v1*v1 + v2*v2)
    lw = 5*speed/speed.max()
    
    x=range(0,nx)
    X,Y=numpy.meshgrid(x,x)
    
    pylab.clf()
    pylab.xlim(0,nx)
    pylab.ylim(0,nx)
    pylab.imshow(rho, cmap=cmap)
    pylab.colorbar()
    #streamplot(X,Y,v2,v1,color='k',linewidth=lw)
    #streamplot(X,Y,v2,v1,color='k')
    pylab.savefig('plot.'+str(i)+'.jpeg')

    return d



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




