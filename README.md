Mickey: Python scripts to tame Pluto
=======================================

![](http://www.queen-of-theme-party-games.com/images/mickey-mouse-party-ideas-21678345.gif) 

Assorted methods and classes to handle and visualize the output of the [Pluto MHD code](https://github.com/black-hole-group/pluto).

# Installation

You have two options to install the module. 

1. Install the module on the system’s python library path: 

    python setup.py install

2. Install the package with a symlink, so that changes to the source files will be immediately available:

    python setup.py develop

This last method is preferred to sync with changes in the repo. You may need to run the last command with `sudo`.


# Simple examples 
	
## To read the simulation output for snapshot 10 (e.g. rho.0010.dbl etc):
	
```python 
import mickey
c=mickey.Pluto(10)
```

The *c* object's attributes are now:
	
- grid: *x1,x2,x3*
- velocities: *v1,v2,v3*
- pressure *p*
- density *rho*
- number of grid cells: *n1,n2,n3*

## Lots of more examples


# TODO

- [ ] compute total mass in volume
- [ ] compute Mdot for a given radius
- [ ] compute energy, a.m. 
- [ ] compute how much mass was lost from the volume due to outflows
- [ ] script to mirror local files to `alphacrucis`, for easy compilation

