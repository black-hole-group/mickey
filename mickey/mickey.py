"""
Methods to read and visualize PLUTO's output.
"""


import pyPLUTO as pp
import numpy
import os
import matplotlib.pyplot as pylab
from scipy import ndimage
import scipy.interpolate








def angleAvg(thArr,arr,theta0,theta1):
	"""
	Given an input 2D array in a spherical polar coordinate basis, 
	and the corresponding coordinates (2D arrays),
	this method computes the angle-average within the angle bounds
	provided. 

	:param thArr: spherical polar angle, 1D, with the corresponding angles where arr is tabulated	
	:param arr: array where the angle-averaging will be carried out (e.g. density)
	:param theta0: initial angle in degrees
	:param theta1: final angle in degrees
	:returns: 1D array with average values as a function of radius

	Example:

	>>> avg=angleAvg(x2,rho,84,96)
	"""
	import nmmn.lsd

	# lower and upper bound for angle average
	th0=theta0*numpy.pi/180.
	th1=theta1*numpy.pi/180.

	# indexes corresponding to the above bounds
	i0=nmmn.lsd.search(th0,thArr)
	i1=nmmn.lsd.search(th1,thArr)

	# Makes sure that the bounds are obeyed
	if thArr[i0]<th0: i0=i0+1
	if thArr[i1]>th1: i1=i1-1

	# angle-average
	arrAvg=numpy.mean(arr.T[i0:i1,:],axis=0)

	# computes the standard deviation of the values
	rhoSd=numpy.std(arr.T[i0:i1,:],axis=0)

	return arrAvg, rhoSd




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
		
	def __init__(self, i=None,*arg,**args):
		# if i is not given, then initialize an empty object
		# otherwise read from the given frame number
		if i is not None:
			d=pp.pload(i,*arg,**args)
			
			# mesh,  and velocities
			'''
			if d.n1>1: 
				self.x1,self.v1,self.n1,self.dx1=d.x1,d.vx1,d.n1,d.dx1
				self.speed=numpy.sqrt(self.v1*self.v1)
			if d.n2>1: 
				self.x2,self.v2,self.n2,self.dx2=d.x2,d.vx2,d.n2,d.dx2
				self.speed=numpy.sqrt(self.v1*self.v1 + self.v2*self.v2)
			if d.n3>1: 
			'''
			self.x1,self.v1,self.n1,self.dx1=d.x1,d.vx1,d.n1,d.dx1
			self.x2,self.v2,self.n2,self.dx2=d.x2,d.vx2,d.n2,d.dx2
			self.x3,self.v3,self.n3,self.dx3=d.x3,d.vx3,d.n3,d.dx3
			self.speed=numpy.sqrt(self.v1*self.v1 + self.v2*self.v2 + self.v3*self.v3)

			# polar coordinates (code units in spherical coords)
			self.r=self.x1
			self.th=-(self.x2-numpy.pi/2.) # spherical angle => polar angle

			# convenient meshgrid arrays
			self.X1,self.X2=numpy.meshgrid(self.x1,self.x2)
			self.DX1,self.DX2=numpy.meshgrid(self.dx1,self.dx2)
			self.R,self.TH=numpy.meshgrid(self.r,self.th)

			# fluid variables
			self.p=d.prs # pressure
			self.rho=d.rho # volume density
			self.getgamma() # gets value of adiabatic index
			self.entropy=numpy.log(self.p/self.rho**self.gamma)
			self.am=self.v3.T*self.X1*numpy.sin(self.X2) # specific a. m., vphi*r*sin(theta)
			self.Be=self.speed.T**2/2.+self.gamma*self.p.T/((self.gamma-1.)*self.rho.T)-1./self.X1	# Bernoulli function
			self.Omega=self.v3.T/self.X1	# angular velocity
 
			# misc. info
			self.t=d.SimTime
			self.pp =d # pypluto object
			self.frame=i
			self.vars=d.vars
			self.geometry=d.geometry

			# sound speed
			#self.soundspeed() # computes numerical cs (no need to specify EoS)
			self.cs=numpy.sqrt(self.gamma*self.p/self.rho)

			# mach number
			if d.n1>1: self.mach1=self.v1/self.cs
			if d.n2>1: self.mach2=self.v2/self.cs
			if d.n3>1: self.mach3=self.v3/self.cs
			self.mach=self.speed/self.cs

			# accretion rates as a function of radius
			self.getmdot() # net accretion rate, self.mdot
			self.getmdotin() # inflow, self.mdotin
			self.getmdotout() # outflow, self.mdotout

			# total mass in computational volume, self.mass
			self.getmass()


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
									



	def mdotr(self,r):
		"""
	Given a certain value of radius, returns the mdot at the nearest
	simulated radius.
		"""
		import nmmn.lsd

		# searches self.r instead of self.x1 to avoid conflicts, e.g.
		# going through a cartesian array rather than a polar one
		i=nmmn.lsd.search(r,self.r)

		return self.mdot[i]





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
		"""
	Computes the net mass accretion rate as a function of radius,

	mdotacc(r) = mdotin(r) + mdotout(r)

	:returns: new attribute mdot, array with the same shape as X1 (radius)
		"""
		# mdot differential
		dmdot=2.*numpy.pi*self.X1**2*self.rho.T*self.v1.T*numpy.sin(self.X2)*self.DX2

		# integrates in theta
		self.mdot=numpy.sum(dmdot, axis=0)




	def getmdotin(self):
		"""
	Computes mass inflow rate as a function of radius. Follows the definition
	of Stone et al. (1999), eq. 10.

	:returns: new attribute mdot, array with the same shape as X1 (radius)
		"""
		# keeps only negative (inflow) radial velocities
		v1=self.v1.copy()
		v1[v1>=0]=0

		# mdot differential
		dmdot=2.*numpy.pi*self.X1**2*self.rho.T*v1.T*numpy.sin(self.X2)*self.DX2

		# integrates in theta
		self.mdotin=numpy.sum(dmdot, axis=0)




	def getmdotout(self):
		"""
	Computes mass outflow rate as a function of radius. Follows the definition
	of Stone et al. (1999), eq. 11.

	:returns: new attribute mdot, array with the same shape as X1 (radius)
		"""
		# keeps only positive (outflow) radial velocities
		v1=self.v1.copy()
		v1[v1<=0]=0

		# mdot differential
		dmdot=2.*numpy.pi*self.X1**2*self.rho.T*v1.T*numpy.sin(self.X2)*self.DX2

		# integrates in theta
		self.mdotout=numpy.sum(dmdot, axis=0)






	def getmass(self):
		"""
	Computes total mass in computational volume.

	:returns: new attribute mass, float
		"""
		# volume differential
		dm=2.*numpy.pi*self.X1**2*self.rho.T*numpy.sin(self.X2)*self.DX1*self.DX2

		# integration
		self.mass=dm.sum()





	def optimalgrid(self):
		"""
	Determines the best resolution when changing coordinate basis from polar
	to cartesian (regrid), to avoid losing information.

	Detailed explanation of procedure available on 
	https://github.com/black-hole-group/group-wiki/blob/master/pluto-analysis-tutorial-02-colormaps-and-regridding.ipynb

	:param :
	:returns:
		"""
		# total area of cartesian grid
		area_car=self.x1[-1]**2 

		# Select the inner parts--with highest resolution--of the polar grid
		# for computing the density of grid elements: 2 cells in r, all cells in
		# theta.
		# 
		# In the future, when we will decrease the angular resolution towards the
		# poles, we will have to choose a subset of theta-elements instead of the 
		# full range, say [n2/4:3/4*n2,0:2]
		X1=self.X1[:,0:2] # r-values
		X2=self.X2[:,0:2] # theta-values
		DX1=self.DX1[:,0:2] # dr-values
		DX2=self.DX2[:,0:2] # dtheta-values

		# variables for inner parts of polar grid
		n_polar=X1.size # number of elements counted
		darea=X1*DX1*DX2 # area elements
		area_polar=darea.sum() # total area for inner grid

		# required number of elements along each direction in cartesian grid
		return int(numpy.sqrt(area_car*n_polar/area_polar))


	

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

		# figures out optimal size of cartesian grid
		if n is None:
			#n=numpy.sqrt(self.x1.size*self.x2.size)*2	# notice the factor of 2
			#n=int(n)
			n=self.optimalgrid()
			print(n)

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

One has to be particularly careful below about using a polar angle
(-pi/2<theta<pi/2) vs a spherical polar angle (0<theta_sph<pi). The
choice can affect some specific transformations.

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

		# figures out optimal size of cartesian grid
		if n is None:
			#n=numpy.sqrt(self.x1.size*self.x2.size)*2	# notice the factor of 2
			#n=int(n)
			n=self.optimalgrid()

		if(gmtry == "SPHERICAL" or gmtry == "CYLINRICAL"):
			xnew=numpy.linspace(0, xlim, n)
			ynew=numpy.linspace(-xlim, xlim, n)
		else:
			xnew=numpy.linspace(-xlim, xlim, n)
			ynew=numpy.linspace(-xlim, xlim, n)

		rho=numpy.zeros((n,n))
		vx=numpy.zeros((n,n))
		vy=numpy.zeros((n,n))
		vz=numpy.zeros((n,n)) # vphi
		p=rho.copy()

		if(gmtry == "SPHERICAL"):
			fastregrid.regrid(xnew, ynew, r, th, self.rho, self.p, self.v1, self.v2, self.v3, rho, p, vx, vy, vz)		
		else: #polar case for bondi
			print("Geometry not supported. Improve the method.")

		# coordinate arrays
		obj.x1,obj.x2=xnew,ynew # cartesian coords, 1D
		obj.X1,obj.X2=numpy.meshgrid(xnew,ynew) # cartesian coords, 2D
		obj.r, obj.th = nmmn.misc.cart2pol(xnew, ynew) # polar coords, 1D
		obj.R, obj.TH = numpy.meshgrid(obj.r,obj.th) # polar coords, 2D
		obj.rsp, obj.thsp = obj.r, numpy.pi/2.-obj.th # spherical polar angle, 1D
		obj.RSP, obj.THSP = numpy.meshgrid(obj.rsp,obj.thsp) # spherical polar coords, 2D

		# velocities
		obj.v1,obj.v2,obj.v3 = vx.T,vy.T,vz.T # Cartesian components
		obj.vr, obj.vth = nmmn.misc.vel_c2p(obj.TH,obj.v1,obj.v2) # polar components
		obj.speed = numpy.sqrt(obj.v1**2+obj.v2**2+obj.v3**3)

		# fluid variables
		obj.gamma=self.gamma
		obj.rho,obj.p=rho.T,p.T
		obj.entropy=numpy.log(obj.p/obj.rho**obj.gamma)
		obj.am=obj.v3*obj.R*numpy.sin(obj.THSP) # specific a. m., vphi*r*sin(theta)
		obj.Be=obj.speed**2/2.+obj.gamma*obj.p/((obj.gamma-1.)*obj.rho)-1./obj.R	# Bernoulli function
		obj.Omega=obj.v3/obj.R	# angular velocity

		# misc info
		obj.regridded=True # flag to tell whether the object was previously regridded
		obj.t=self.t
		obj.frame=self.frame
		obj.mdot=self.mdot
		obj.mass=self.mass

		return obj





