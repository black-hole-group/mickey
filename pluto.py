"""
Methods to read and visualize PLUTO's output.
"""


import pyPLUTO as pp
#import pylab, numpy
import numpy
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



def search(xref, x):
   """
Search for the element in an array x with the value nearest xref.
Piece of code based on http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array

>>> i=search(xref, x)

:param xref: input number, array or list of reference values
:param x: input array
:returns: index of the x-elements with values nearest to xref:
   """
   if numpy.size(xref)==1:
      i=(numpy.abs(x-xref)).argmin()
   else:
      i=[]

      for y in xref:
         i.append( (numpy.abs(x-y)).argmin() )

   return i


def pol2cart(r, phi):
   """
Converts from polar to cartesian coordinates.

>>> x,y=pol2cart(r,phi)
   """
   x = r * numpy.cos(phi)
   y = r * numpy.sin(phi)
   return x, y


def cart2pol(x, y):
   """
Converts from cartesian to polar coordinates.

>>> r,t=cart2pol(x,y)
   """
   r = numpy.sqrt(x**2 + y**2)
   #t = numpy.arctan2(y, x)
   t = numpy.arctan2(x, y)
   return r, t






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
   - bx1, bx2, bx3


   To read the simulation output for frame 10:

   >>> import pluto
   >>> p=pluto.Pluto(10)

   Plots density field:

   >>> p.snap()
   """

   def __init__(self, i=0):
     if(i < 0):
      self.x1=0
      self.v1,self.n1=0,0
      self.x2= 0
      self.v2,self.n2=0,0
      self.x3=0
      self.v3,self.n3=0,0
      self.speed=0
      self.bx3 = 0

      self.p=0
      self.rho=0
      self.pp=0 # pypluto object

      self.frame=-1


     else:
      d=pp.pload(i)

      # when getting xi below, assumes uniform grid
      if d.n1>1:
         self.x1=d.x1
         self.v1,self.n1=d.vx1,d.n1
         self.speed=numpy.sqrt(self.v1*self.v1)
#         self.bx1 = d.bx1
      if d.n2>1:
         self.x2=d.x2
         self.v2,self.n2=d.vx2,d.n2
         self.speed=numpy.sqrt(self.v1*self.v1 + self.v2*self.v2)
#         self.bx2 = d.bx2
      if d.n3>1:
         self.x3=d.x3
         self.v3,self.n3=d.vx3,d.n3
         self.speed=numpy.sqrt(self.v1*self.v1 + self.v2*self.v2 + self.v3*self.v3)
         self.bx3 = d.bx3

      self.p=d.prs
      self.rho=d.rho
      self.dp =  numpy.gradient(d.prs)
      self.drho = numpy.gradient(d.rho)
      self.pp=d # pypluto object
      self.frame=i
      # this is probably incorrect
      #self.Mdot=-4.*numpy.pi*self.x1**2*self.rho*self.v1



   def snap(self,n=20,lim=10,rhomax = 2,stream = 'n',mag = 'n',var=None,hor=None):
      """
Creates snapshot of 2D simulation generated in any coordinates.

:param n: Number of uniform divisions in x and y for the quiver plot
:param lim: The limits which the graph will be plotted (from -lim to lim)
:param var: variable to be plotted. If not specified, assumes rho
:param hor: plots circle at inner boundary radius with radius=hor. If None, no circle

>>> p=pluto.Pluto(10)
>>> p.snap(10,p.p)
      """
#      import seaborn
#      seaborn.set_style({"axes.grid": False})
#      cmap=seaborn.cubehelix_palette(light=1, as_cmap=True)

      d = self.pp
      lw = 5*self.speed/self.speed.max()
      I = pp.Image()

      pylab.clf()
      if(d.geometry=='POLAR'):
          I.pldisplay(d, numpy.log(d.rho),x1=d.x1,x2=d.x2,
                label1='x',label2='$y$',title=r'Density $\rho$ ',
                cbar=(True,'vertical'),polar=[True,True],vmin=-9,vmax=rhomax) #polar automatic conversion =D
          #obj = self.pol2cart(n,lim)
          pylab.title("t = %.2f" % (d.SimTime))
          #pylab.quiver(obj.x1,obj.x2,obj.v1,obj.v2,color='k')
          pylab.xlim(-lim,lim)
          pylab.ylim(-lim,lim)
          print ("Done i= %i" % self.frame)

      if(d.geometry=='SPHERICAL'):
          I.pldisplay(d, numpy.log(d.rho),x1=d.x1,x2=d.x2,
                label1='R',label2='$z$',title=r'Density $\rho$ ',
                cbar=(True,'vertical'),polar=[True,False],vmin=-5,vmax=rhomax) #polar automatic conversion =D
          #obj = self.pol2cart(n,lim)
          pylab.title("t = %.2f  " % (float(d.SimTime)/6.28318530717) + "$\\rho_{max}$ = %.3f" % numpy.max(self.pp.rho))
          #pylab.quiver(obj.x1,obj.x2,obj.v1,obj.v2,color='k')
          pylab.xlim(0,2*lim)
          pylab.ylim(-lim,lim)
          pylab.tight_layout()
          print "Done i= %i" % self.frame
      else:
         I.pldisplay(d, numpy.log(d.rho),x1=d.x1,x2=d.x2,
                     label1='r',label2='$\phi$',lw=lw,title=r'Density $\rho$ [Torus]',
                cbar=(True,'vertical'),vmin=-9,vmax=0) #polar automatic conversion =D
         obj = self.cart(n,lim)
#         self.plot_grid()
         pylab.title("t = %.2f" % d.SimTime)
         if stream == 'y':
            if(mag == 'y'):
                pylab.streamplot(obj.x1,obj.x2,obj.bx1,obj.bx2,color='k')
            else:
                pylab.streamplot(obj.x1,obj.x2,obj.v1,obj.v2,color='k')
         else:
            if(mag == 'y'):
                pylab.quiver(obj.x1,obj.x2,obj.bx1,obj.bx2,color='k')
            else:
                pylab.quiver(obj.x1,obj.x2,obj.v1,obj.v2,color='k')
         pylab.xlim(self.x1.min(),2*lim)
         pylab.ylim(-lim,lim)
      if hor!=None:
         circle=pylab.Circle((0,0),hor,color='b')
         pylab.gca().add_artist(circle)
      #pylab.streamplot(self.x1,self.x2,self.v2,self.v1,color='k')

      pylab.tight_layout()
      pylab.savefig('plot.'+str(self.frame)+'.png',dpi=400)
#      pylab.show()


   def contour_newgrid(self, n=200, xlim = None,rhocut = None):
      """
Creates a new object with variables in cartesian coordinates.
Useful if the object was created by PLUTO in polar coordinates.

:param n: New number of elements n^2.
:param xlim: Boundary for the plot and the grid
:param rhocut: Variable used if you want to put a lower limit to the contours

      """
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
                rnew,thnew=cart2pol(xnew[i],ynew[j])
                # position in old array
                iref=search(rnew, r)
                jref=search(thnew, th)
                if(self.rho[iref,jref] < rhocut): #for contours with a low limit
                   rho[i,j] = rhocut
                else:
                   rho[j,i]=self.rho[iref,jref]
                p[j,i]=self.p[iref,jref]
                vx[j,i]=self.v1[iref,jref]
                vy[j,i]=self.v1[iref,jref]

            else: #polar case for bondi
                # position in old array
                iref=search(xnew[i], r)
                jref=search(ynew[j], th)
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
Function for contour plotting. It recieves some parameters to foward to respective functions.
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
    d = stone_fig5(Ni,Nf)
    n = 5
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
    generic_plot(d.x1,numpy.log10(rhop),subplt=221,xlabel="Radius",
                ylabel="$\\rho$",xlim=[0.01,1],ylim=[0.1,1],xscale='log',
                yscale='log',color='b')
    if(files != None):
        pylab.plot(files[0].T[0],files[0].T[1],'k')
    ######Pressure#######
    generic_plot(d.x1,numpy.log10(prsp),subplt=222,xlabel="Radius",
                ylabel="$P$",xlim=[0.01,1],ylim=[0.01,10],xscale='log',
                yscale='log',color='b')
    if(files != None):
        pylab.plot(files[1].T[0],files[1].T[1],'k')
    ######Radial Velocity#######
    generic_plot(d.x1,numpy.abs(vradp),subplt=223,xlabel="Radius",
                ylabel="$v_r$",xlim=None,ylim=None,xscale='log',
                yscale='log',color='b')
    if(files != None):
        pylab.plot(files[2].T[0],files[2].T[1],'k')
    ######Angular Velocity#######
    generic_plot(d.x1,numpy.abs(vphp),subplt=224,xlabel="Radius",
                ylabel="$v_\\phi$",xlim=[0.01,1],ylim=[0.01,1],xscale='log',
                yscale='log',color='b')
    if(files != None):
        pylab.plot(files[3].T[0],files[3].T[1],'k')
    #############
    pylab.tight_layout()
    pylab.savefig("sph_ana" + ".png",dpi=dpi)
    pylab.show()
    pylab.clf()
    print "Done sph_plot"




###################################################
def sum_pclass(soma,aux):
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
def normalize(soma,k):
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
    k = 0
    soma = Pluto(Ni)
    for i in range(Ni+1,Nf+1):
        aux = Pluto(i)
        sum_pclass(soma,aux)
        k += 1
    normalize(soma,k)
    return soma
###################################################
