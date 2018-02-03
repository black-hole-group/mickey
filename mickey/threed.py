

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
