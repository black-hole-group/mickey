def render(file):
    d=nmmn.grmhd.Raishin()
    d.vtk(file)
    var=d.regrid(d.rho) # <===== specify variable to be plotted
    var=nmmn.grmhd.fixminus(var) # removes negative values
    
    # output filename
    root=re.search(r'\w+\d+',file).group()
    idfile=int(re.search(r'\d+',file).group())
    
    pylab.clf()
    pylab.imshow(numpy.log10(var),extent=[d.xc1d[0],d.xc1d[-1],d.yc1d[0],d.yc1d[-1]], cmap=mycmap, vmin=-6, vmax=0)
    pylab.xlim(0,30)
    pylab.ylim(-20,20)
    pylab.axes().set_aspect('equal')
    pylab.colorbar()
    pylab.title('$t= $'+str(idfile*10)+' $GM/c^3$') # careful with normalization
    
    # black hole
    circle2=pylab.Circle((0,0),1,color='k')
    pylab.gca().add_artist(circle2)
    
    #pylab.savefig(root+'.png',transparent=True,dpi=150)