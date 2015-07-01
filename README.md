PLUTO Tools
============

Assorted methods and classes to handle and visualize the output of the PLUTO MHD code.
	
To read the simulation output for snapshot 10 (e.g. rho.0010.dbl etc):
	
```python 
import pluto
c=pluto.Pluto(10)
```

The *p* object's attributes are now:
	
- grid: x1,x2,x3
- velocities: v1,v2,v3
- pressure p
- density rho
- number of grid cells: n1,n2,n3


Plots density field in cartesian coordinates:

```python	
c.snap()
```

or if you don't want to define an object:

```python	
pluto.Pluto(10).snap()
```

Plots a density field for snapshot 50 which was generated in polar coordinates, "regridded" with a 400x400 cartesian grid:

```python	
p=pluto.Pluto(50).pol2cart(400)
p.snap()
```
or
```python	
pluto.Pluto(50).pol2cart(400).snap()
```

More examples coming soon.