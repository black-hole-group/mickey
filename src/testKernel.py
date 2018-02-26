import pylab
import mickey.mickey
import mickey.plot
import numpy as np
import numpy

v=mickey.mickey.Pluto(107)

import nmmn.lsd, nmmn.misc


# number of desired points for regridding
n=1500


# creates copy of current object which will have the new
# coordinates
obj=mickey.mickey.Pluto() # empty pluto object

# r, theta
r=v.x1
th=-(v.x2-numpy.pi/2.) # spherical angle => polar angle
xlim = v.x1.max()
gmtry = v.pp.geometry

xnew=numpy.linspace(0, xlim, n)
ynew=numpy.linspace(-xlim, xlim, n)

rho=numpy.zeros((n,n))
vx=numpy.zeros((n,n))
vy=numpy.zeros((n,n))
vz=numpy.zeros((n,n))
p=rho.copy()


import pyopencl as cl  


# In[75]:

platforms=cl.get_platforms()
devices=platforms[0].get_devices(cl.device_type.CPU)
context=cl.Context([devices[0]])
#context = cl.create_some_context()
queue = cl.CommandQueue(context)
mf = cl.mem_flags


# ## Host variables
# 
# Define host arrays with appropriate precision, suffix `_h`. First the input arrays

# In[76]:


# xnew, ynew, r, th, v.rho, v.p, v.v1, v.v2, v.v3
xnew_h = xnew.astype(np.float32)
ynew_h = ynew.astype(np.float32)
r_h = r.astype(np.float32)
th_h = th.astype(np.float32)
rhoin_h = v.rho.astype(np.float32)
pin_h = v.p.astype(np.float32)
v1in_h = v.v1.astype(np.float32)
v2in_h = v.v2.astype(np.float32)
v3in_h = v.v3.astype(np.float32)


# then the output arrays

# In[77]:


# rho, p, vx, vy
rho_h = rho.astype(np.float32)
p_h = p.astype(np.float32)
vx_h = vx.astype(np.float32)
vy_h = vy.astype(np.float32)
vz_h = vz.astype(np.float32)


# ## Device variables
# 
# Buffers, suffix `_d`. Input arrays

# In[78]:


xnew_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xnew_h)
ynew_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ynew_h)
r_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=r_h)
th_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=th_h)
rhoin_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rhoin_h)
pin_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pin_h)
v1in_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=v1in_h)
v2in_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=v2in_h)
v3in_d = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=v3in_h)


# Output arrays

# In[79]:


# rho, p, vx, vy
rho_d = cl.Buffer(context, mf.WRITE_ONLY, rho_h.nbytes)
p_d = cl.Buffer(context, mf.WRITE_ONLY, rho_h.nbytes)
vx_d = cl.Buffer(context, mf.WRITE_ONLY, rho_h.nbytes)
vy_d = cl.Buffer(context, mf.WRITE_ONLY, rho_h.nbytes)
vz_d = cl.Buffer(context, mf.WRITE_ONLY, rho_h.nbytes)


# ## Kernel definition

# In[87]:


# ## Execute kernel

# In[100]:


kernel=open('fastregrid.cl').read()

program = cl.Program(context, kernel).build()


# In[101]:

program.regrid(queue, rho_h.shape, None, numpy.int32(xnew.size), xnew_d, numpy.int32(ynew.size), ynew_d, numpy.int32(r.size), r_d, numpy.int32(th.size), th_d, rhoin_d, pin_d, v1in_d, v2in_d, v3in_d, rho_d, p_d, vx_d, vy_d, vz_d)


# ## Gathers output

# In[90]:


cl.enqueue_copy(queue, rho_h, rho_d)
cl.enqueue_copy(queue, p_h, p_d)
cl.enqueue_copy(queue, vx_h, vx_d)
cl.enqueue_copy(queue, vy_h, vy_d)
cl.enqueue_copy(queue, vz_h, vz_d)

