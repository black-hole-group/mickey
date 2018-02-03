import pylab
import numpy

def render(obj,lim=None, bh=None, file=False):
    """
Plots the density scalar field for a Pluto simulation.

:param obj: pluto object created with Mickey, storing the arrays for the simulation you want to visualize
:param lim: specify bounding box for plot
    """
    # gets arrays in a cartesian grid, stored in a new object p
    p=obj.regrid()
        
    pylab.clf()
    pylab.imshow(numpy.log10(p.rho),extent=[d.x1[0],d.x1[-1],d.x2[0],d.x2[-1]], #vmin=-6, vmax=0)

    if lim is None:
        pylab.xlim(0,2*lim)
        pylab.ylim(-lim,lim)

    pylab.axes().set_aspect('equal')
    pylab.colorbar()
    pylab.title('$t= $'+str( (obj.pp.SimTime/(2.*numpy.pi)) ) + "$\\rho_{\\rm max}$ = " + str (p.rho.max()))
    
    # black hole
    if bh is not None:
        circle2=pylab.Circle((0,0),bh,color='k')
        pylab.gca().add_artist(circle2)
    
    if file is True:
        pylab.savefig('plot.'+str(obj.frame)+'.png',transparent=True,dpi=300)