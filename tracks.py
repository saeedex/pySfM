from config import *
from views import *


class Track:
    def __init__(self):
        self.str = np.empty((0, 6), float)
        self.valid = np.empty(0, bool)
        self.removed = np.empty(0, bool)

        self.views = []
        self.idx = []
        self.res = []
        self.inliers = []

    # matching current keyframe (kf) features with connected keyframes features (conviews)
    def matchmap(self, views, kf, config):
        conviews = np.zeros(0, int)
        for i in range(len(views.graph.conviews[kf])):
            mf = views.graph.conviews[kf][i]
            fidx, ratio = Track.match2views(views, kf, mf, config)

            if ratio > 0.1:
                conviews = np.concatenate([np.array([mf], int), conviews])
                kmatched = views.matched[kf][fidx[:, 0]]
                mmatched = views.matched[mf][fidx[:, 1]]

                # if matched feature is not previously matched create new tracks
                nindices = np.multiply(~mmatched, ~kmatched).flatten()
                nkidx = fidx[nindices, 0]
                nmidx = fidx[nindices, 1]
                ntracks = np.arange(len(self.str), len(self.str) + len(nkidx))
                self.create(nkidx)
                views.addtrack(self, ntracks, mf, nmidx)
                views.addtrack(self, ntracks, kf, nkidx)
                self.addview(ntracks, mf, nmidx)
                self.addview(ntracks, kf, nkidx)

                # if matched feature is previously matched create update existing tracks
                cindices = np.multiply(mmatched, ~kmatched).flatten()
                ckidx = fidx[cindices, 0]
                cmidx = fidx[cindices, 1]
                mtracks = views.trackids[mf][cmidx]
                views.addtrack(self, mtracks, kf, ckidx)
                self.addview(mtracks, kf, ckidx)

                cindices = np.multiply(~mmatched, kmatched).flatten()
                ckidx = fidx[cindices, 0]
                cmidx = fidx[cindices, 1]
                ktracks = views.trackids[kf][ckidx]
                views.addtrack(self, ktracks, mf, cmidx)
                self.addview(ktracks, mf, cmidx)

                # check and handle duplicates
                dindices = np.multiply(mmatched, kmatched).flatten()
                dkidx = fidx[dindices, 0]
                dmidx = fidx[dindices, 1]
                ktracks = views.trackids[kf][dkidx]
                mtracks = views.trackids[mf][dmidx]
                check = np.not_equal(ktracks, mtracks)

                if np.sum(check):
                    ktracks = ktracks[check]
                    mtracks = mtracks[check]
                    views.rmvtrack(self, ktracks)
                    self.rmvview(ktracks)
                    views.mrgtrack(self, mtracks, ktracks)
                    self.mrgview(mtracks, ktracks)

        views.graph.conviews[kf] = conviews

    def create(self, kidx):
        self.str = np.concatenate([self.str, np.zeros([len(kidx), 6], float)])
        self.valid = np.concatenate([self.valid, np.zeros(len(kidx), bool)])
        self.removed = np.concatenate([self.removed, np.zeros(len(kidx), bool)])

        for i in range(len(kidx)):
            self.views.append([])
            self.idx.append([])
            self.res.append([])
            self.inliers.append([])

    def addview(self, tracks, kf, kidx):
        for j in range(len(tracks)):
            i = tracks[j]
            self.views[i] = np.concatenate([np.array(self.views[i], int), np.array([kf], int)])
            self.idx[i] = np.concatenate([np.array(self.idx[i], int), np.array([kidx[j]], int)])
            self.res[i] = np.concatenate([np.array(self.res[i], float), np.array([0], float)])
            self.inliers[i] = np.concatenate([np.array(self.inliers[i], bool), np.array([0], bool)])

    def rmvview(self, tracks):
        for j in range(len(tracks)):
            i = tracks[j]
            self.str[i, :] = 0
            self.valid[i] = False
            self.removed[i] = True

    def mrgview(self, tracks, dtracks):
        for j in range(len(tracks)):
            i = tracks[j]
            self.views[i] = np.concatenate([self.views[i], self.views[dtracks[j]]])
            self.idx[i] = np.concatenate([self.idx[i], self.idx[dtracks[j]]])
            self.res[i] = np.concatenate([self.res[i], self.res[dtracks[j]]])
            self.inliers[i] = np.concatenate([self.inliers[i], self.inliers[dtracks[j]]])

            self.views[i], ia = np.unique(self.views[i], return_index=True)
            self.idx[i] = self.idx[i][ia]
            self.res[i] = self.res[i][ia]
            self.inliers[i] = self.inliers[i][ia]

    def evaluate(self, views, tracks, config):
        for j in range(len(tracks)):
            i = tracks[j]

            for k in range(len(self.views[i])):
                kf = self.views[i][k]
                idx = self.idx[i][k]
                kobs = np.reshape(views.obs[kf][idx, :], (1, 2))
                point = np.reshape(self.str[i, :3], (1, 3))
                iprj, iz = Track.project(views.K[kf], views.pose[kf], point)

                res = np.sum(np.square(iprj - kobs))
                depth = iz
                if (res < config.opt.threshold) & (depth < config.dataset.maxz) & (depth > config.dataset.minz):
                    self.inliers[i][k] = True

    def actview(self, tracks):
        for j in range(len(tracks)):
            i = tracks[j]
            ninliers = np.sum(self.inliers[i])
            self.valid[i] = ninliers > 0.6 * self.inliers[i].shape[0]

    # triangulate new tracks
    def triangulate(self, views, kf, config):
        kproj = np.dot(views.K[kf], views.pose[kf])
        nkidx = np.multiply(views.matched[kf], ~views.tracked[kf]).flatten()
        ntracks = views.trackids[kf][nkidx]
        nkobs = views.obs[kf][nkidx, :]

        for j in range(len(ntracks)):
            i = ntracks[j]
            for k in range(len(self.views[i]) - 1):
                mf = self.views[i][k]
                nmidx = self.idx[i][k]
                nmobs = views.obs[mf][nmidx, :]
                mproj = views.K[mf] @ views.pose[mf]
                point = cv2.triangulatePoints(mproj, kproj, nmobs, nkobs[j, :])
                point = point / point[3]
                self.str[ntracks[j], :3] = point[:3].T

        self.getcolor(ntracks, views.color[kf], nkobs)
        self.evaluate(views, ntracks, config)
        self.actview(ntracks)
        views.acttrack(self, ntracks)
        return ntracks

    # get RGB color of observations
    def getcolor(self, ntracks, img, obs):
        # read image
        img = img.astype(np.float32)
        img /= 255.
        r = img[:, :, 2]
        g = img[:, :, 1]
        b = img[:, :, 0]
        r = r.flatten()
        g = g.flatten()
        b = b.flatten()

        # get pixel colors
        obs = np.round(obs)
        obs = obs.astype(int)
        idx = (obs[:, 0] - 1) * img.shape[0] + obs[:, 1]
        idx = idx.astype(int)
        b = b[idx]
        g = g[idx]
        r = r[idx]
        self.str[ntracks, 3:] = np.array([r, g, b]).T

    # project a point onto image frame
    @staticmethod
    def project(K, pose, points):
        points = np.concatenate([points, np.ones([points.shape[0], 1])], axis=1)
        points = np.dot(pose, points.T)
        iz = points[2, :]
        points = np.divide(points, points[2, :])
        points = np.dot(K, points)
        iprj = points[:2, :].T
        return iprj, iz

    # compute a reprojection error of a point
    @staticmethod
    def residual(K, pose, points, obs):
        iprj, _ = Track.project(K, pose, points)
        res = np.sum(np.square(iprj - obs), axis=1)
        return res

    # match features in keyframes kf and mf
    @staticmethod
    def match2views(views, kf, mf, config):
        mdsc = views.dsc[mf]
        kdsc = views.dsc[kf]
        matches = config.sparse.matcher.knnMatch(kdsc, mdsc, k=2)
        inlier_matches = []

        fidx = np.empty([0, 2])
        for m, n in matches:
            if m.distance < config.sparse.threshold * n.distance:
                inlier_matches.append([m])
                fidx = np.append(fidx, np.array([[m.queryIdx, m.trainIdx]]), axis=0)
        fidx = fidx.astype(int)
        fidx = np.unique(fidx, axis=0)
        ratio = fidx.shape[0] / views.matched[kf].shape[0]
        return fidx, ratio