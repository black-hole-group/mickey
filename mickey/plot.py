import pylab
import numpy

def density(obj,lim=None, bh=None, file=False, n=None, *arg, **args):
    """
Plots the density scalar field for a Pluto simulation. Offers several options
for convenience. 

:param obj: pluto object created with Mickey, storing the arrays for the simulation you want to visualize
:param lim: specify bounding box for plot 
:param bh: plots circle centered on the origin to illustrate the inner boundary. Can be specified as either a float or a list [float,string] ([radius,color string], e.g. [1,'w'])
:param file: should we output a graphic file with the plot?
:param n: DEPRECATED, number of cartesian grid points per dimension, total number of points = n^2 

Example:

>>> mickey.plot.density(mickey.mickey.Pluto(107),lim=1, bh=[0.1,'w'], vmin=-4, vmax=0)

will plot the 107th snapshot, performing regridding if necessary.
    """
    # gets arrays in a cartesian grid, stored in a new object p
    #if hasattr(obj, 'regridded'):
    #    p=obj
    #else:
    #    if n is None:
    #        p=obj.regrid(400)
    #    else:
    #        p=obj.regrid(n)
        
    pylab.clf()
    #pylab.imshow(numpy.log10(p.rho),extent=[p.x1[0],p.x1[-1],p.x2[0],p.x2[-1]], *arg, **args)
    pylab.pcolormesh(obj.X, obj.Y, log10(obj.rho.T), *arg, **args)

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






def densityn(i,lim=None, bh=None, file=False, n=None, *arg, **args):
    """
Plots the density scalar field for a Pluto simulation. Offers several options
for convenience. 

:param i: integer specifying the simulation frame you desire to plot
:param lim: specify bounding box for plot 
:param bh: plots circle centered on the origin to illustrate the inner boundary. Can be specified as either a float or a list [float,string] ([radius,color string], e.g. [1,'w'])
:param file: should we output a graphic file with the plot?
:param n: number of cartesian grid points per dimension, total number of points = n^2 

Example:

>>> mickey.plot.density(107,lim=1, bh=[0.1,'w'], vmin=-4, vmax=0)

will plot the 107th snapshot, performing regridding if necessary.
    """
    from . import mickey

    p=mickey.Pluto(i, stdout=False)

    # crops arrays
    rho, X1, X2 = nmmn.lsd.crop(obj.v1, obj.X1, obj.X2, x0,x1,y0,y1,all=True)

    # gets arrays in a cartesian grid, stored in a new object p
    if n is None:
        c=p.regridFast(1500)
    else:
        c=p.regridFast(n) 
        
    pylab.clf()
    pylab.imshow(numpy.log10(p.rho),extent=[p.x1[0],p.x1[-1],p.x2[0],p.x2[-1]], *arg, **args)

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












def mesh(obj):
    """
Plot the computational mesh grid.

:param obj: pluto object created with Mickey, storing the arrays for the simulation you want to visualize
    """
    import nmmn.misc

    # makes sure this is not a regridded object
    if hasattr(obj, 'pp'): 
        if obj.pp.geometry=='SPHERICAL':
            # EXECUTE COMMAND ONLY FOR THE SUPPORTED GEOMETRY

            theta=-(obj.x2-numpy.pi/2.) # spherical angle => polar angle
            radius=obj.x1

            for th in theta:
                tt=th*numpy.ones_like(radius)
                x,y=nmmn.misc.pol2cart(radius,tt)
                pylab.plot(x,y,'gray')

            for r in radius:
                rr=r*numpy.ones_like(theta)
                x,y=nmmn.misc.pol2cart(rr,theta)
                pylab.plot(x,y,'gray')

            pylab.xlabel('$x$')
            pylab.ylabel('$y$')
            pylab.axes().set_aspect('equal')
    else:
        print("Geometry not currently supported. Implement this.")



def streamplot(obj, x0, x1, y0, y1, *arg, **args):
    """
Fast version of matplotlib's streamplot method. Instead of computing streamlines
for the whole domain of the arrays, this method crops the array to the given
[x0,x1,y0,y1] range. This yields a great speedup.

For a demonstration of the technique, please check out the jupyter notebook
below:
https://github.com/black-hole-group/group-wiki/blob/master/pluto-analysis-tutorial-02-colormaps-and-regridding.ipynb

:param obj: pluto object created with Mickey, storing the arrays for the simulation you want to visualize
:param x0, x1: specify the plotting range for x
:param y0, y1: specify the plotting range for y

Any extra arguments provided will be passed to matplotlib's streamplot.

Example

>>> p=mickey.mickey.Pluto(10)
>>> mickey.plot.streamplot(p, 0, 2, -1, 1, density=1.5)
    """
    import nmmn.lsd

    # crops arrays
    v1crop, X1crop, X2crop = nmmn.lsd.crop(obj.v1, obj.X1, obj.X2, x0,x1,y0,y1,all=True)
    v2crop = nmmn.lsd.crop(obj.v2, obj.X1, obj.X2, x0,x1,y0,y1)

    pylab.streamplot(X1crop, X2crop, v1crop, v2crop, *arg, **args)

