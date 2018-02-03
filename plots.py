



def contours(self,N,lim,plot_flag='y'):
				"""
		PROBABLY BUGGY. BE CAREFUL!

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








def generic_plot(X,Y,**kwargs):
		"""
		This function is made so the user can call an plot
		using just one line
		"""
		if(kwargs['subplt'] != None):
				pylab.subplot(kwargs['subplt'])
		if(kwargs['xlabel'] != None):
				pylab.xlabel(kwargs['xlabel'])
		if(kwargs['ylabel'] != None):
				pylab.ylabel(kwargs['ylabel'])
		if(kwargs['xlim'] != None):
				pylab.xlim(kwargs['xlim'])
		if(kwargs['ylim'] != None):
				pylab.ylim(kwargs['ylim'])
		if(kwargs['xscale'] != None):
				pylab.xscale(kwargs['xscale'])
		if(kwargs['yscale'] != None):
				pylab.yscale(kwargs['yscale'])
		pylab.plot(X,Y,kwargs['color'])





def sph_analisys(Ni,Nf,files=None):
		"""
Function to make plots 5 and 6 of stone et al 99.
It also plots the stone version of this plot if you extracted the data
from the pdf. **Not yet working properly**

:param Ni: Starting snapshot
:param Nf: Ending snapshot
:param files: A path to files that contain the data from stone

.. todo:: fix normalization and units
.. todo:: allow to specify *ti, tf* instead of *Ni*, *Nf*
		"""
		d = stone_fig5(Ni,Nf)
		# opening angle (theta) in degrees that will be used to make averaging
		# around the equator
		n = 4
		thmin = (90-n) * numpy.pi / 180.
		thmax = (90+n) * numpy.pi / 180.
		dpi = 400
		######Setting vectors for plot#######
		rhop = numpy.zeros(d.n1)
		prsp = numpy.zeros(d.n1)
		vphp = numpy.zeros(d.n1)
		vradp = numpy.zeros(d.n1)
		######Loop time!#######
		for i in range(d.n1):
				rho = numpy.zeros(d.n2)
				prs = numpy.zeros(d.n2)
				vph = numpy.zeros(d.n2)
				vrad = numpy.zeros(d.n2)
				for j in range(d.n2):
						if( d.x2[j] > thmin ):
								rho[j] = d.rho[i,j]
								prs[j] = d.p[i,j]
								vph[j] = d.v2[i,j]
								vrad[j] = d.v1[i,j]
						if(d.x2[j] > thmax):
								break
				rhop[i] = numpy.sum(rho)
				prsp[i] = numpy.sum(prs)
				vphp[i] = numpy.sum(vph)
				vradp[i] = numpy.sum(vrad)
		######Density#######
		generic_plot(d.x1,rhop,subplt=221,xlabel="Radius",
								ylabel="$\\rho$",xlim=None,ylim=None,xscale='log',
								yscale='log',color='b')
		if(files != None):
				pylab.plot(files[0].T[0],files[0].T[1],'k')
		######Pressure#######
		generic_plot(d.x1,prsp,subplt=222,xlabel="Radius",
								ylabel="$P$",xlim=None,ylim=None,xscale='log',
								yscale='log',color='b')
		if(files != None):
				pylab.plot(files[1].T[0],files[1].T[1],'k')
		######Radial Velocity#######
		generic_plot(d.x1,vphp,subplt=223,xlabel="Radius",
								ylabel="$v_\\phi$",xlim=[0.01,1],ylim=None,xscale='log',
								yscale='log',color='b')
		if(files != None):
				pylab.plot(files[2].T[0],files[2].T[1],'k')
		######Angular Velocity#######
		generic_plot(d.x1,numpy.abs(vradp),subplt=224,xlabel="Radius",
								ylabel="abs$(v_r)$",xlim=[0.01,10],ylim=[0.01,1],xscale='log',
								yscale='log',color='b')
		if(files != None):
				pylab.plot(files[3].T[0],files[3].T[1],'k')
		#############
		pylab.tight_layout()
		pylab.savefig("sph_ana" + ".png",dpi=dpi)
		pylab.show()
		pylab.clf()
		print "Done sph_plot"


def fig3_stone(ti,t_tot,N_snap):
	"""
Plots figure 3 from stone
:param ti: time for computation
:param t_tot: total time of the simulation
:param N_snap: total number of snapshots
	"""
	frame = int(t_tot/(ti*N_snap))
	d = Pluto(frame)

	I = pp.Image()
	xx,yy = I.getPolarData(d.pp,d.pp.x2)
	gamma = 5./3
	pylab.subplot(131)
	var = numpy.transpose(numpy.log10(d.rho))
	pylab.pcolormesh(xx,yy,var,cmap="Oranges")
	pylab.axis([0,2,-1,1])
	pylab.subplot(132)
	var = numpy.transpose(numpy.log(d.p/(d.rho**gamma)))
	pylab.pcolormesh(xx,yy,var,cmap="Oranges")
	pylab.axis([0,2,-1,1])
	pylab.tick_params(
		axis='y',          # changes apply to the x-axis
		which='both',      # both major and minor ticks are affected
		left='off',      # ticks along the bottom edge are off
		right='off',         # ticks along the top edge are off
		labelleft='off') # labels along the bottom edge are off
	pylab.subplot(133)
	var = numpy.copy(d.rho)
	for i in range(d.n1):
		for j in range(d.n2):
			var[i,j] = d.rho[i,j] * (d.v2[i,j]*xx[j,i] - numpy.sqrt(xx[j,i]))
	var = numpy.transpose(var)
	pylab.pcolormesh(xx,yy,var,cmap="Oranges")
	pylab.axis([0,2,-1,1])
	pylab.tick_params(
		axis='y',          # changes apply to the x-axis
		which='both',      # both major and minor ticks are affected
		left='off',      # ticks along the bottom edge are off
		right='off',         # ticks along the top edge are off
		labelleft='off') # labels along the bottom edge are off
	pylab.show()
	pylab.clf()


###################################################
def sum_pclass(soma,aux):
		"""
Receives two pluto classes and sum aux into soma.
In other words, sums all variables for two states
of the simulation.

:param soma: Pluto class that will be added aux
:param aux: Pluto class to be added in soma
		"""
		soma.x1 += aux.x1
		soma.v1 += aux.v1
		if(soma.pp.n2>1):
				soma.x2 += aux.x2
				soma.v2 += aux.v2
		if(soma.pp.n3>1):
				soma.x3 += aux.x3
				soma.v3 += aux.v3
		soma.p += aux.p
		soma.rho += aux.rho
###################################################
def mean(soma,k):
		"""
Receives two pluto classes and computes the mean of
all variables associated.

:param soma: Pluto class to be normalized
:param k: number to normalize
		"""
		soma.x1 /= k
		soma.v1 /= k
		if(soma.pp.n2>1):
				soma.x2 /= k
				soma.v2 /= k
		if(soma.pp.n3>1):
				soma.x3 /= k
				soma.v3 /= k
		soma.p /= k
		soma.rho /= k
###################################################
def stone_fig5(Ni,Nf):
		"""
Function to plot stone fig5.

:param Ni: Starting snapshot
:param Nf: Ending snapshot

.. warning:: Defintly not working. Cf. normalization.
		"""
		k = 0
		soma = Pluto(Ni)
		for i in range(Ni+1,Nf+1):
				aux = Pluto(i)
				sum_pclass(soma,aux)
				k += 1
		mean(soma,k)
		return soma
###################################################

def torus_plot(args):
		i,n,lim,rhomax,stream,mag = args[0],args[1],args[2],args[3],args[4],args[5]
		D = Pluto(i)
		D.snap(n,lim,rhomax,stream,mag)
		del D

def contour_plot(args):
		D,N,lim = Pluto(args[0]),args[1],args[2]
		D.contours(N,lim)
		print "Done i = %d" %D.frame
		del D

def run_fig2(Rb,Nf,t_tot):
		Njump = 4
		File = open("mdot_data.dat",'w')
		Mdot = numpy.zeros(Nf+1)
		Mdot5 = numpy.zeros(Nf+1)
		Mdot10 = numpy.zeros(Nf+1)
		t = numpy.zeros(Nf+1)
		Rfin=0
		for i in range(Nf):
				if(i%Njump == 0):
						pp = Pluto(i)
						if(i==0):
								print pp.rho.sum()
						X,Y = numpy.meshgrid(pp.x1,pp.x2) # X == r, Y == theta
						M = numpy.abs(2*numpy.pi*X*X)
						M *= numpy.abs(pp.rho*pp.v1)
						M *=  numpy.abs(numpy.sin(Y)*abs(pp.x2[0]-pp.x2[1]))
						stop
						Mdot[i] = numpy.sum(M[0,:])
						k = search(5*Rb,pp.x1)
						Mdot5[i] = numpy.sum(M[k,:])
						k = search(10*Rb,pp.x1)
						Mdot10[i] = numpy.sum(M[k,:])
						t[i] = pp.pp.SimTime / (2*numpy.pi)
						File.write(str(t[i]) + " " +str(Mdot[i]) + " " +str(Mdot5[i]) + " " +str(Mdot10[i]) + "\n")
						if(pp.pp.SimTime > (t_tot-2*numpy.pi)):
								Rfin += Mdot[i]
		#pylab.plot(numpy.linspace(0,t/(2*numpy.pi),Nf/Njump + 1),R,'k')
		#data = numpy.loadtxt("mdot_data.dat")
		#t,Mdot,Mdot5,Mdot10 = data[0].T,data[1].T,data[2].T,data[3].T
		pylab.plot(t,Mdot,'k.')
		pylab.plot(t,Mdot5,'r.')
		pylab.plot(t,Mdot10,'b.')
		pylab.yscale('log')
		pylab.xlabel("Orbits")
		pylab.ylabel("Mdot")
#    pylab.ylim(1e-4,1e-2)
		pylab.savefig("teste.png")
		pylab.show()
		print("Macc = %e" % Rfin)

def integ(pp):
		dr = numpy.zeros(pp.n1)
		dth = numpy.zeros(pp.n2)
		for i in range(pp.n1-1):
				dr[i] = numpy.abs(pp.x1[i]-pp.x1[i+1])
		for i in range(pp.n2-1):
				dth[i] = numpy.abs(pp.x2[i]-pp.x2[i+1])
		dX,dY = numpy.meshgrid(dth,dr,sparse=True)
		X,Y = numpy.meshgrid(pp.x2,pp.x1,sparse=True)

		M = 2*numpy.pi*pp.rho*X**2 * numpy.sin(Y) * dX * dY
		soma = numpy.sum(M)
		print soma
		return soma

def interpolate(pp):
		X,Y = numpy.meshgrid(pp.x2,pp.x1)
		f = sp.interpolate.interp2d(pp.rho,X,Y,kind='linear')
		pylab.imshow(f(pp.x1,pp.x2))
		pylab.show()

def call_multiprocess(f,N,w_dir,N_procs):
	 os.chdir(w_dir)
	 p = mp.Pool(N_procs)
	 aux = []
	 for i in range(N):
			 aux.append(i)
	 p.map(f,aux)
	 os.system("sh density.sh 60 density")
	 os.chdir("..")



def run_torus(w_dir,N_snap,xlim,Nmin=0):
		os.chdir(w_dir)
		bnd = []
		n = 400
		rhomax = 0
		stream = 'y'
		mag = 'y'
		for i in range(Nmin,N_snap+1):
				bnd.append([i,n,xlim,rhomax,stream,mag])
		#p = mp.Pool(mp.cpu_count())
		proc = 8
		p = mp.Pool(proc)
		p.map(torus_plot,bnd)
		#density_plot([10,100,0.1])
		os.system("sh density.sh 60 density")
		os.chdir("..")

def run_contour(w_dir,N_snap,N,Nmin=0):
		os.chdir(w_dir)
		Nmax = 200
		bnd = []
		lim = 2
		proc = 8
		p = mp.Pool(proc)
		for i in range(Nmin,N_snap+1):
				if(i%Nmax == 0):
						p.map(contour_plot,bnd)
						del bnd
						bnd = []
				bnd.append([i,N,lim])
		p.map(contour_plot,bnd)
		os.system("sh contour.sh 60 contour")
		os.chdir("..")


def run_fig5(w_dir,ni,nf):
		###########
		Nplots = 4
		files = []
		for i in range(Nplots):
				files.append(numpy.loadtxt(w_dir+"../"+"digt_data/plot_stone"+str(i+1)+".csv",delimiter=','))
		###########
		os.chdir(w_dir)
		sph_analisys(ni,nf,files)
		#stone_fig2(0.01,nf)
		os.chdir("..")

def plotB1(bnd):#,cs_inf,rho_inf):
	 import numpy as np
	 import matplotlib.pylab as plt
	 """
			Create a jpeg for a given snapshot using B1 parameters
			where bnd is a vector with a loaded bondi class
	 """
	 r = np.linspace(0.1,5)
	 color = 'r'
	 s = 5.0
	 marker = '.'
	 gamma = 5./3
	 C1 = 10.1
	 C2 = 0.01
	 C3 = 1.4
	 t = 1.53
	 Nmax = 153
	 ################
	 plt.subplot(221)
	 plt.title("t = %.2f" % (bnd.pp.SimTime))
	 for j in range(bnd.n2):
			 for k in range(bnd.n3):
							 plt.scatter(bnd.x1,bnd.rho[:,j,k],color=color,s=s,marker=marker,label="Simulation")
	 plt.plot(r/10,C1*np.power(r,-3/2.),label="Model")
	 data = np.loadtxt("B1_rho.csv",delimiter=',')
	 plt.plot(data.T[0],data.T[1],'k',label="Ruffert 94")
	 plt.xlabel("Radius $R$")
	 plt.ylabel("Density $\\rho$")
	 plt.ylim(1,200)
	 plt.xlim(1e-2,5e0)
	 plt.yscale('log')
	 plt.xscale('log')
	 plt.legend(loc=0,prop={'size':10})
	 plt.grid(True)
	 ################
	 plt.subplot(222)
	 csv = np.sqrt(gamma*bnd.p/bnd.rho)
	 for j in range(bnd.n2):
			 for k in range(bnd.n3):
					 plt.scatter(bnd.x1,map(abs,bnd.v1[:,j,k]/csv[:,j,k]),color=color,s=s,marker=marker)
	 plt.plot(r,C2*np.power(r,-2))
	 data = np.loadtxt("B1_mach.csv",delimiter=',')
	 plt.plot(data.T[0],data.T[1],'k')
	 plt.xlabel("Radius $R$")
	 plt.ylabel("$ v/c $")
	 plt.ylim(1e-1,2)
	 plt.xlim(1e-2,5e0)
	 plt.legend(loc=0,prop={'size':10})
	 plt.yscale('log')
	 plt.xscale('log')
	 plt.grid(True)
	 ################
	 plt.subplot(223)
	 #acc = 4*np.pi*bnd.R_b**2*bnd.rho_inf*bnd.cs_inf
	 #bnd_acc= np.linspace(acc,acc,len(bnd.r))
#   plt.scatter(bnd.x2,bnd.flux,color=color,s=s,marker=marker)
	 #print '%.2f' % bnd.flux[bnd.indx]
#   plt.scatter(bnd.r,bnd_acc,color='b',s=s,marker=marker)
	 plt.xlabel("Radius $R$")
	 plt.ylabel("Accretion Rate $\dot M$")
	 plt.xlim(1e-2,5e0)
	 plt.ylim(0,10)
	 #plt.yscale('log')
	 plt.xscale('log')
	 plt.legend(loc=0,prop={'size':10})
	 plt.grid(True)
	 ################
	 plt.subplot(224)
	 for j in range(bnd.n2):
			 for k in range(bnd.n3):
					 plt.scatter(bnd.x1,csv[:,j,k],color=color,s=s,marker=marker)
	 plt.plot(r/10,C3*np.power(r,-1/2.))
	 data = np.loadtxt("B1_sspeed.csv",delimiter=',')
	 plt.plot(data.T[0],data.T[1],'k')
	 plt.xlabel("Radius $R$")
	 plt.ylabel("Sound Speed $c_s$")
	 plt.ylim(1,10)
	 plt.xlim(1e-2,5)
	 plt.yscale('log')
	 plt.xscale('log')
	 plt.grid(True)
	 plt.legend(loc=0,prop={'size':10})
	 ################
	 plt.tight_layout()
	 plt.savefig("rho_vel" + str(bnd.frame) + ".jpeg",dpi=400)
	 plt.clf()

	 print "Done i=",bnd.frame


def run_mayavi(i):
		from mayavi.mlab import quiver3d

		d = pluto(i)

		quiver3D(d.rho)