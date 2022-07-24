from tracks import *

class vis3d:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(visible=True)
        ctr = self.vis.get_render_option()
        ctr.point_size = 1
        ctr.background_color = np.array([0, 0, 0])

    def update(self, *args):
        Track = args[0]

        if len(args) > 1:
            points = Track.str[args[1], :]
            valid = Track.valid[args[1]]
        else:
            points = Track.str
            valid = Track.valid

        points = points[valid, :]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(points[:, 3:])

        self.vis.add_geometry(pcd)
        ctr = self.vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        self.vis.poll_events()
        self.vis.update_renderer()

    def animate(self, *args):
        Track = args[0]

        if len(args) > 1:
            points = Track.str[args[1], :]
            valid = Track.valid[args[1]]
        else:
            points = Track.str
            valid = Track.valid

        points = points[valid, :]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(points[:, 3:])

        def rotate_view(vis):
            ctr = vis.get_view_control()
            ctr.rotate(1.0, 0.0)
            ctr = vis.get_render_option()
            ctr.point_size = 1
            ctr.background_color = np.array([0, 0, 0])
            return False
        o3d.visualization.draw_geometries_with_animation_callback([pcd], rotate_view)

    def destroy(self):
        self.vis.destroy_window()

class vis2d:
    def observation(*args):
        radius = 2
        img = args[0]
        obs = args[1]
        color = args[2]

        for p in obs:
            cv2.circle(img, (int(p[0]), int(p[1])), radius, color, -1)
        return img

    def reprojection(*args):
        Track = args[0]
        Views = args[1]
        Images = args[2]
        kf = args[3]
        config = args[4]

        if config.sparse.display:
            kimg = Images.color[kf]
            kobs = Views.sparse.obs[kf][Views.sparse.tracked[kf], :]
            tracks = Views.sparse.tracks[kf][Views.sparse.tracked[kf]]

            valid = np.array(Track.valid)[tracks]
            removed = np.array(Track.removed)[tracks]
            inliers = np.multiply(valid, ~removed).flatten()

            kobs = kobs[inliers, :]
            tracks = tracks[inliers]
            points = Track.str[tracks, :3]
            prj, _ = track.project(Views.camera.K[kf], Views.camera.pose[kf], points)
            vis2d.observation(kimg, prj, [0, 0, 255])
            vis2d.observation(kimg, kobs, [255, 0, 0])
            cv2.imshow("Observations", kimg)
            cv2.waitKey(1)