from config import *
from iomodule import *
from tracks import *
from visualization import *
from views import *
#import pyboof as pb

# configuring
config = config()
Images = images()
Images.load(config)
Views = views(Images, config)
Track = track()

# load camera intrinsics
Views.camera.loadintr(Images, config)
#Views.camera.loadposes(Images, config)

# detect features
Views.sparse.detect(Images, config)

# sparse reconstruction
Vis3d = vis3d() # this is for visualization

for mf in range(len(Images.color)-1):
    kf = mf + 1
    Views.graph.update(kf)

    # sparse matching
    Views = Track.matchmap(Views, kf, config)

    # pose initialization
    Views.camera.computepose(Views, Track, kf)

    # triangulation
    Views, ntracks = Track.triangulate(Views, Images, kf, config)

    # optimization

    # visualization
    vis2d.reprojection(Track, Views, Images, kf, config)
    Vis3d.update(Track, ntracks)

# visualization
Vis3d.destroy()
Vis3d.animate(Track)