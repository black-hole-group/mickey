"""
Methods to read and visualize PLUTO's output.
"""


import pyPLUTO as pp
import numpy
import os
import matplotlib.pyplot as pylab
from scipy import ndimage
import multiprocessing as mp
import scipy.interpolate
from mpl_toolkits.mplot3d import Axes3D




def movie(fname="movie.avi"):
	 """
3D movie generation.
	 """
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
			#snap(i)
			peixe.animate(amount=i)






#### 3d routines


def cutplane(i):
	 """
Snapshot of 3d cartesian simulation, generating cut planes.

i : index corresponding to frame you want to plot
	 """
	 import mayavi.mlab as mlab

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
	 import mayavi.mlab as mlab

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
	
	>>> import mickey
	>>> p=mickey.Pluto(10)
	
	Plots density field:
	
	>>> p.snap()
	"""
		
	def __init__(self, i=None,gamma=1.66666,*arg,**args):
		# if i is not given, then initialize an empty object
		# otherwise read from the given frame number
		if i is not None:
			d=pp.pload(i,*arg,**args)
			
			# mesh,  and velocities
			if d.n1>1: 
				self.x1,self.v1,self.n1,self.dx1=d.x1,d.vx1,d.n1,d.dx1
				self.speed=numpy.sqrt(self.v1*self.v1)
			if d.n2>1: 
				self.x2,self.v2,self.n2,self.dx2=d.x2,d.vx2,d.n2,d.dx2
				self.speed=numpy.sqrt(self.v1*self.v1 + self.v2*self.v2)
			if d.n3>1: 
				self.x3,self.v3,self.n3,self.dx3=d.x3,d.vx3,d.n3,d.dx3
				self.speed=numpy.sqrt(self.v1*self.v1 + self.v2*self.v2 + self.v3*self.v3)

			# pressure
			self.p=d.prs
			self.p_grad = numpy.gradient(d.prs)
			# volume density
			self.rho=d.rho 
			self.rho_grad = numpy.gradient(d.rho)
			# time
			self.t=d.SimTime

			# misc. info
			self.pp =d # pypluto object
			self.frame=i
			self.vars=d.vars
			self.geometry=d.geometry

			# sound speed
			self.getgamma() # gets value of adiabatic index
			#self.soundspeed() # computes numerical cs (no need to specify EoS)
			self.cs=numpy.sqrt(self.gamma*self.p/self.rho)

			# mach number
			if d.n1>1: self.mach1=self.v1/self.cs
			if d.n2>1: self.mach2=self.v2/self.cs
			if d.n3>1: self.mach3=self.v3/self.cs
			self.mach=self.speed/self.cs

			# accretion rates
			#self.getmdot()	# => self.mdot


	def getgamma(self):
		"""
	Gets value of gamma from "pluto.ini".
		"""
		try:
			f = open("pluto.ini","r")
		except IOError as e: 
			print(e)

		for line in f:
				if 'GAMMA' in line:
						s=line.split() # splits string divided by whitespaces
						self.gamma=float(s[1])
									




	def soundspeed(self,smooth=None):
		"""
	Compute cs=sqrt(dP/drho) which is valid for a general EoS.

	1. Uses the data itself to find out P(rho)
	2. Removes repeated values and does a linear interpolation of P(rho)
	3. Gets the derivative dP/drho
	4. Computes the cs array
		"""
		import nmmn.lsd

		# P=P(rho), 
		# i.e. gives you the pressure as a function of density
		# =====================
		# but first: NEED TO DISCARD REPEATED VALUES in P and rho
		rho=[]	# unique values of rho
		p=[]	# unique corresponding values of P 
		# orders arrays of simulation (which have repeated values)
		i=nmmn.lsd.sortindex(self.rho.flatten())
		rhosim=self.rho.flatten()[i]
		psim=self.p.flatten()[i]
		# after this loop, you will get arrays with unique elements
		for j,x in enumerate(rhosim):
			if x not in rho:
				rho.append(x)
				p.append(psim[j])

		# creates interpolated arrays for P and rho
		# cf. http://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html#spline-interpolation-in-1-d-procedural-interpolate-splxxx
		import scipy.interpolate
		if smooth==None:
			pfun = scipy.interpolate.splrep(rho, p)
		else:
			pfun = scipy.interpolate.splrep(rho, p,s=smooth)

		# calculates dP/drho in the same grid as the sim
		pdiff=scipy.interpolate.splev(self.rho,pfun,der=1)

		# sound speed
		self.csnum=numpy.sqrt(pdiff)



	def getmdot(self):
		# compute mass accretion rate valid for any accretion flow
		# right now, this is only valid for polar 2d sims

		# arrays convenient for vectorization
		r,th = numpy.meshgrid(self.x1,self.x2)
		dr,dth = numpy.meshgrid(self.dx1,self.dx2)

		#x=2.*pi*d['rho'][2,:]*d['x1'][2]**2*sin(d['x2'])*d['v1'][2,:]*d['dx2']
		dmdot=2.*numpy.pi*self.rho*r.T**2*numpy.sin(th)*self.v1*dth.T

		self.mdot=dmdot.sum(1) # sums along phi axis



	

	def snap(self,var=None,hor=None,rhomax=None,lim=None,stream=False,mag=False,file=False):
		"""
Renders the density field for a given frame.
Input: 2D simulation generated in any coordinates.

:param n: Number of uniform divisions in x and y for the quiver plot
:param lim: The limits which the graph will be plotted (from -lim to lim)
:param var: variable to be plotted. If not specified, assumes rho
:param hor: plots circle at inner boundary radius with radius=hor. If None, no circle

>>> p=pluto.Pluto(10)
>>> p.snap(10,p.p)
		"""
		# import seaborn
		# seaborn.set_style({"axes.grid": False})
		# cmap=seaborn.cubehelix_palette(light=1, as_cmap=True)

		d = self.pp
		lw = 5*self.speed/self.speed.max()
		I = pp.Image()

		# should we fix the max density? Useful for animations to avoid the spurious
		# flickering effect
		if rhomax is None:
			rhomax=self.rho.max()

		pylab.clf()

		# Depending on the geometry, calls the appropriate function
		# to perform coordinate transformations
		cmap = 'Oranges'
		if(d.geometry=='POLAR'):
				# here pldisplay does the cartesian conversion
				I.pldisplay(d, numpy.log10(d.rho),x1=d.x1,x2=d.x2,
							label1='x',label2='$y$',title='Density $\rho$ ',
							cbar=(True,'vertical'),polar=[True,True],vmin=-9,vmax=rhomax,cmesh=cmap) #polar automatic conversion =D
				#obj = self.pol2cart(n,lim)
				pylab.title("t = %.2f" % (d.SimTime))
				#pylab.quiver(obj.x1,obj.x2,obj.v1,obj.v2,color='k')
				if lim is not None:
					pylab.xlim(-lim,lim)
					pylab.ylim(-lim,lim)
				print ("Done i= %i" % self.frame)

		if(d.geometry=='SPHERICAL'):
				I.pldisplay(d, numpy.log10(d.rho),x1=d.x1,x2=d.x2,
							label1='R',label2='$z$',title=r'Density $\rho$ ',
							cbar=(True,'vertical'),polar=[True,False],vmin=-5,vmax=rhomax,cmesh=cmap) #polar automatic conversion =D
				#obj = self.pol2cart(n,lim)
				pylab.title("t = %.2f  " % (float(d.SimTime)/6.28318530717) + "$\\rho_{\\rm max}$ = %.3f" % numpy.max(self.pp.rho))
				#pylab.quiver(obj.x1,obj.x2,obj.v1,obj.v2,color='k')
				if lim is not None:
					pylab.xlim(0,2*lim)
					pylab.ylim(-lim,lim)

				pylab.tight_layout()
				print("Done i= %i" % self.frame)
		else:
			I.pldisplay(d, numpy.log10(d.rho),x1=d.x1,x2=d.x2,
									 label1='r',label2='$\phi$',lw=lw,title=r'Density $\rho$ [Torus]',
							cbar=(True,'vertical'),vmin=-9,vmax=0,cmesh=cmap) #polar automatic conversion =D
			obj = self.cart(n,lim)
#         self.plot_grid()
			pylab.title("t = %.2f" % d.SimTime)
			if stream == True:
					if(mag == True):
							pylab.streamplot(obj.x1,obj.x2,obj.bx1,obj.bx2,color='k')
					else:
							pylab.streamplot(obj.x1,obj.x2,obj.v1,obj.v2,color='k')
			else:
					if(mag == True):
							pylab.quiver(obj.x1,obj.x2,obj.bx1,obj.bx2,color='k')
					else:
							pylab.quiver(obj.x1,obj.x2,obj.v1,obj.v2,color='k')

			if lim is not None:
				pylab.xlim(self.x1.min(),2*lim)
				pylab.ylim(-lim,lim)

		if hor!=None:
			 circle=pylab.Circle((0,0),hor,color='k')
			 pylab.gca().add_artist(circle)
		#pylab.streamplot(self.x1,self.x2,self.v2,self.v1,color='k')

		pylab.tight_layout()

		if file is True:
			pylab.savefig('plot.'+str(self.frame)+'.png',dpi=400)





	def regrid(self, n=None, xlim = None):
			"""
Transforms a mesh in arbitrary coordinates (e.g. nonuniform elements)
into a uniform grid in the same coordinates.

:param n: New number of elements n^2. If None, figures out by itself
:param xlim: Boundary for the plot and the grid
			"""
			import nmmn.lsd, nmmn.misc

			# creates copy of current object which will have the new
			# coordinates
			obj=Pluto() # empty pluto object

			# r, theta
			r=self.x1
			th=-(self.x2-numpy.pi/2.) # spherical angle => polar angle
			if(xlim == None):
					xlim = self.x1.max()
			gmtry = self.pp.geometry

			# figures out size of cartesian grid
			if n is None:
				n=numpy.sqrt(self.x1.size*self.x2.size)*2	# notice the factor of 2
				n=int(n)

			if(gmtry == "SPHERICAL" or gmtry == "CYLINRICAL"):
				xnew=numpy.linspace(0, xlim, n)
				ynew=numpy.linspace(-xlim, xlim, n)
			else:
				xnew=numpy.linspace(-xlim, xlim, n)
				ynew=numpy.linspace(-xlim, xlim, n)

			rho=numpy.zeros((n,n))
			vx=numpy.zeros((n,n))
			vy=numpy.zeros((n,n))
			p=rho.copy()

			# goes through new array
			for i in range(xnew.size):
				for j in range(ynew.size):
						if(gmtry == "SPHERICAL"):
								rnew,thnew=nmmn.misc.cart2pol(xnew[i],ynew[j])
								# position in old array
								iref=nmmn.lsd.search(rnew, r)
								jref=nmmn.lsd.search(thnew, th)

								rho[j,i]=self.rho[iref,jref]
								p[j,i]=self.p[iref,jref]
								vx[j,i]=self.speed[iref,jref]*numpy.cos(thnew)
								vy[j,i]=self.speed[iref,jref]*numpy.sin(thnew)

						else: #polar case for bondi
								# position in old array
								iref=nmmn.lsd.search(xnew[i], r)
								jref=nmmn.lsd.search(ynew[j], th)
								rho[i,j]=self.rho[iref,jref]
								p[i,j]=self.p[iref,jref]
								vx[i,j]=self.v1[iref,jref] * numpy.cos(thnew)
								vy[i,j]=self.v1[iref,jref] * numpy.sin(thnew)

		#set new variables to null object
			obj.x1,obj.x2=xnew,ynew
			obj.rho,obj.p=rho,p
			obj.v1,obj.v2 = vx,vy

			return obj











	def contour_newgrid(self, n=200, xlim = None,rhocut = None):
			"""
Transforms a mesh in arbitrary coordinates (e.g. nonuniform elements)
into a uniform grid in the same coordinates.

:param n: New number of elements n^2.
:param xlim: Boundary for the plot and the grid
:param rhocut: Variable used if you want to put a lower limit to the contours
			"""
			import nmmn.lsd, nmmn.misc

			# creates copy of current object which will have the new
			# coordinates
			obj=Pluto(-1) #null pluto object
			if(rhocut == None):
					rhocut = -1

			# r, theta
			r,th=self.x1,self.x2
			if(xlim == None):
					xlim = self.x1.max()
			gmtry = self.pp.geometry

			if(gmtry == "SPHERICAL" or smtry == "CYLINRICAL"):
					xnew=numpy.linspace(0, xlim, n)
					ynew=numpy.linspace(-xlim, xlim, n)
			else:
					xnew=numpy.linspace(-xlim, xlim, n)
					ynew=numpy.linspace(-xlim, xlim, n)

			rho=numpy.zeros((n,n))
			vx=numpy.zeros((n,n))
			vy=numpy.zeros((n,n))
			p=rho.copy()

			# goes through new array
			for i in range(xnew.size):
				for j in range(ynew.size):
						if(gmrty == "SPHERICAL" or gmrty == "CYLINDRICAL"):
								rnew,thnew=nmmn.misc.cart2pol(xnew[i],ynew[j])
								# position in old array
								iref=nmmn.lsd.search(rnew, r)
								jref=nmmn.lsd.search(thnew, th)
								if(self.rho[iref,jref] < rhocut): #for contours with a low limit
									 rho[i,j] = rhocut
								else:
									 rho[j,i]=self.rho[iref,jref]
								p[j,i]=self.p[iref,jref]
								vx[j,i]=self.v1[iref,jref]
								vy[j,i]=self.v1[iref,jref]

						else: #polar case for bondi
								# position in old array
								iref=nmmn.lsd.search(xnew[i], r)
								jref=nmmn.lsd.search(ynew[j], th)
								rho[i,j]=self.rho[iref,jref]
								p[i,j]=self.p[iref,jref]
								vx[i,j]=self.v1[iref,jref] * numpy.cos(thnew)
								vy[i,j]=self.v1[iref,jref] * numpy.sin(thnew)

		#set new variables to null object
			obj.x1,obj.x2=xnew,ynew
			obj.rho,obj.p=rho,p
			obj.v1,obj.v2 = vx,vy

			return obj





	def contours(self,N,lim,plot_flag='y'):
				"""
Function for contour plotting. It can plot also the density map,
setting plot_flag to 'y'

:param N: Size of grid
:param lim: is the plot limit
:param plot_flag: control the plot of the density map
				"""
				rhocut = None
				if (self.pp.geometry == "SPHERICAL"):
						rhocut = 5e-5
				obj = self.contour_newgrid(N,lim,rhocut)
				xi,yi,zi = obj.x1,obj.x2,numpy.log10(obj.rho)

				#plot the density map
				pylab.clf()
				d = self.pp
				if(plot_flag == 'y'):
						I = pp.Image()
						I.pldisplay(d, numpy.log(d.rho),x1=d.x1,x2=d.x2,
										label1='x',label2='$y$',title=r'Density $\rho$ ',
										cbar=(True,'vertical'),polar=[True,False],cmap='YlOrBr',vmin=-4,vmax=0) #polar automatic conversion =D
				#plot contour
				pylab.rcParams['contour.negative_linestyle'] = 'solid' #set positive and negative contour as solid
				pylab.contour(xi,yi,zi,20,colors='k')
				pylab.title("t = %.2f  " % (float(d.SimTime)/6.28318530717) + "$\\rho_{max}$ = %.3f" % numpy.max(self.pp.rho))
				pylab.xlim(0,lim)
				pylab.ylim(-lim/2.,lim/2.)

				pylab.savefig("contour_plot"+str(self.frame)+".png",dpi=300)
				pylab.clf()




