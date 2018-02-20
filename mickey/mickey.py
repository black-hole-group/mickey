"""
Methods to read and visualize PLUTO's output.
"""


import pyPLUTO as pp
import numpy
import os
import matplotlib.pyplot as pylab
from scipy import ndimage
import scipy.interpolate




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









class Pluto:
	"""
	Class that defines data objects imported from PLUTO.
	
	The object's attributes are:
	
	- x1,x2,x3
	- v1,v2,v3
	- pressure p
	- rho
	- n1,n2,n3

	To supress the message when reading a binary file, use stdout=False.

	
	To read the simulation output for frame 10:
	
	>>> import mickey
	>>> p=mickey.mickey.Pluto(10)
	
	:param stdout: if False, suppresses message mentioning the file that was processed
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

			# convenient meshgrid arrays
			self.X1,self.X2=numpy.meshgrid(self.x1,self.x2)
			self.DX1,self.DX2=numpy.meshgrid(self.dx1,self.dx2)

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
Renders the density field for a given frame using pldisplay in pypluto.

Input: 2D simulation generated in any coordinates.

:param n: Number of uniform divisions in x and y for the quiver plot
:param lim: The limits which the graph will be plotted (from -lim to lim)
:param var: variable to be plotted. If not specified, assumes rho
:param hor: plots circle at inner boundary radius with radius=hor. If None, no circle

>>> p=pluto.Pluto(10)
>>> p.snap(10,p.p)
		"""
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

		#pylab.tight_layout()

		if file is True:
			pylab.savefig('plot.'+str(self.frame)+'.png',dpi=400)





	def regrid(self, n=None, xlim = None):
		"""
Transforms a mesh in arbitrary coordinates (e.g. nonuniform elements)
into a uniform grid in the same coordinates.

:param n: New number of elements n^2. If None, figures out by itself
:param xlim: Boundary for the plot and the grid

.. todo:: speed this up with C, the loops slow this down in python.
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

		# *****BOTTLENECK*****
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
						# careful with cartesian conversion for vectors
						vx[j,i],vy[j,i]=nmmn.misc.vel_p2c(thnew,self.v1[iref,jref],self.v2[iref,jref])

					else: #polar case for bondi
						print("Geometry not supported. Improve the method.")
		# *****END BOTTLENECK*****

	#set new variables to null object
		obj.x1,obj.x2=xnew,ynew
		obj.rho,obj.p=rho,p
		obj.v1,obj.v2 = vx,vy
		obj.regridded=True # flag to tell whether the object was previously regridded
		obj.t=self.t
		obj.frame=self.frame
		obj.speed=numpy.sqrt(vx*vx + vy*vy)
		obj.X1,obj.X2=numpy.meshgrid(xnew,ynew)

		return obj




	def regridFast(self, n=None, xlim = None):
		"""
Transforms a mesh in arbitrary coordinates (e.g. nonuniform elements)
into a uniform grid in the same coordinates. Uses a C function to 
speed things up. 

:param n: New number of elements n^2. If None, figures out by itself
:param xlim: Boundary for the plot and the grid
		"""
		import nmmn.lsd, nmmn.misc

		# C function for fast regridding. Make sure you compile it first
		# with make
		import fastregrid

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

		if(gmtry == "SPHERICAL"):
			fastregrid.regrid(xnew, ynew, r, th, self.rho, self.p, self.v1, self.v2, rho, p, vx, vy)		
		else: #polar case for bondi
			print("Geometry not supported. Improve the method.")

	#set new variables to null object
		obj.x1,obj.x2=xnew,ynew
		obj.rho,obj.p=rho,p
		obj.v1,obj.v2 = vx,vy
		obj.regridded=True # flag to tell whether the object was previously regridded
		obj.t=self.t
		obj.frame=self.frame
		obj.speed=numpy.sqrt(vx*vx + vy*vy)
		obj.X1,obj.X2=numpy.meshgrid(xnew,ynew)

		return obj





