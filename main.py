from config import *
from tracks import *
from visualization import *
from views import *

# configuring
config = config()
views = views()
tracks = Track()


# load
views.loadimages(config)
views.initcameras(config)
views.loadintr(config)
views.loadposes(config)

# detect features
views.detect(config)

# sparse reconstruction
v3d = v3d()

for mf in range(len(views.color) - 1):
    kf = mf + 1
    views.graph.update(kf)

    # sparse matching
    tracks.matchmap(views, kf, config)

    # pose initialization
    #views.computepose(tracks, kf)

    # triangulation
    ntracks = tracks.triangulate(views, kf, config)

    # visualization
    v2d.reprojection(tracks, views, kf, config)
    v3d.update(tracks, ntracks)

# visualization
v3d.destroy()
v3d.animate(tracks)