import pylab
import numpy

def density(obj,lim=None, bh=None, file=False, n=None):
    """
Plots the density scalar field for a Pluto simulation.

:param obj: pluto object created with Mickey, storing the arrays for the simulation you want to visualize
:param lim: specify bounding box for plot as [radius,color string], e.g. [1,'w']
    """
    # gets arrays in a cartesian grid, stored in a new object p
    if hasattr(obj, 'regridded'):
        p=obj
    else:
        if n is None:
            p=obj.regrid(400)
        else:
            p=obj.regrid(n)
        
    pylab.clf()
    pylab.imshow(numpy.log10(p.rho),extent=[p.x1[0],p.x1[-1],p.x2[0],p.x2[-1]]) #vmin=-6, vmax=0)

    if lim is not None:
        pylab.xlim(0,2*lim)
        pylab.ylim(-lim,lim)

    pylab.xlabel('$x$')
    pylab.ylabel('$y$')
    pylab.axes().set_aspect('equal')
    cbar=pylab.colorbar()
    cbar.set_label("$\log \\rho$")
    pylab.title('$t= $'+str( round(obj.t/(2.*numpy.pi),2) ) + ", $\\rho_{\\rm max}$ = " + str(round(p.rho.max(),2)) )
    
    # black hole
    if bh is not None:
        if numpy.size(bh)==1:
            circle2=pylab.Circle((0,0),bh,color='k')
        else:
            circle2=pylab.Circle((0,0),bh[0],color=bh[1])
            
        pylab.gca().add_artist(circle2)
    
    if file is True:
        pylab.savefig('plot.'+str(obj.frame)+'.png',transparent=True,dpi=300)