from config import *


class views:
    def __init__(self):
        self.dense = []
        self.graph = self.graph()

        # images
        self.color = []
        self.gray = []

        # sparse matching
        self.dsc = []
        self.obs = []
        self.matched = []
        self.tracked = []
        self.trackids = []

        # camera parameters
        self.pose = []
        self.K = []

    def initcameras(self, config):
        # init poses and intrinsics
        opose = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
        f = np.max(self.gray[0].shape) * config.dataset.intscale
        params = [f, f, self.gray[0].shape[1] / 2, self.gray[0].shape[0] / 2]
        K = np.array([[params[0], 0, params[2]], [0, params[1], params[3]], [0, 0, 1]])
        for kf in range(len(self.color)):
            self.pose.append(opose)
            self.K.append(K)

    def loadimages(self, config):
        os.chdir(config.dataset.path)
        for i in glob.glob("*.jpg"):
            img = cv2.imread(config.dataset.path + i)
            img = cv2.resize(img, (int(img.shape[1] / config.dataset.imresize),
                                   int(img.shape[0] / config.dataset.imresize)))
            gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.color.append(img)
            self.gray.append(gimg)

            if config.dataset.display:
                cv2.imshow('sample image', img)
                cv2.waitKey(1)

    def loadintr(self, config):
        datafile = open(config.dataset.path + "/" + 'cameraIntrinsic.txt')
        params = list()

        for line in datafile:
            strs = line.split(' ', line.count(' '))
            for strvar in strs:
                params.append(float(strvar))

        if self.color[1].shape[0] > 640:
            params = params * 3

        K = np.array([[params[0], 0, params[2]], [0, params[1], params[3]], [0, 0, 1]])
        K[:2, :] = K[:2, :] / config.dataset.imresize

        for kf in range(len(self.color)):
            self.K[kf] = K

    def loadposes(self, config):
        datafile = open(config.dataset.path + "/" + 'cameras.out')
        data = list()

        for line in datafile:
            linedata = list()
            strs = line.split(' ', line.count(' '))
            for strvar in strs:
                linedata.append(float(strvar))
            data.append(linedata)

        for kf in range(len(self.color)):
            ext = np.array(data[kf * 5 + 1:kf * 5 + 5])
            R = ext[:3, :]
            t = np.reshape(ext[3, :], (3, 1))

            pose = np.hstack((R, t))
            pose[1:, :] = -pose[1:, :]
            self.pose[kf] = pose

    def detect(self, config):
        for img in self.color:
            kpts, dscs = config.sparse.detector.detectAndCompute(img, None)
            obs = np.float32([kpts[i].pt for i in range(len(kpts))])

            self.dsc.append(dscs)
            self.obs.append(obs)
            self.matched.append(np.zeros(obs.shape[0], dtype=bool))
            self.tracked.append(np.zeros(obs.shape[0], dtype=bool))
            self.trackids.append(np.zeros(obs.shape[0], dtype=int))

    def addtrack(self, tracks, trackids, kf, kidx):
        self.matched[kf][kidx] = True
        self.tracked[kf][kidx] = tracks.valid[trackids]
        self.trackids[kf][kidx] = trackids

    def rmvtrack(self, tracks, trackids):
        for j in range(len(trackids)):
            i = trackids[j]

            for k in range(len(tracks.views[i])):
                kf = tracks.views[i][k]
                idx = tracks.idx[i][k]
                self.trackids[kf][idx] = 0
                self.tracked[kf][idx] = False
                self.matched[kf][idx] = False

    def mrgtrack(self, tracks, trackids, dtrackids):
        for j in range(len(trackids)):
            i = trackids[j]
            for k in range(len(tracks.views[dtrackids[j]])):
                kf = tracks.views[dtrackids[j]][k]
                kidx = tracks.idx[dtrackids[j]][k]
                self.trackids[kf][kidx] = i

    def acttrack(self, tracks, trackids):
        for j in range(len(trackids)):
            i = trackids[j]

            for k in range(len(tracks.views[i])):
                kf = tracks.views[i][k]
                idx = tracks.idx[i][k]
                self.tracked[kf][idx] = tracks.valid[i]

    def computepose(self, tracks, kf):
        if kf == 1:
            mf = kf - 1
            trackids = self.trackids[kf][self.matched[kf]]
            fidx = np.array(tracks.idx)[trackids]
            midx = fidx[:, 0]
            kidx = fidx[:, 1]

            mobs = self.obs[mf][midx, :]
            kobs = self.obs[kf][kidx, :]

            # Fundamental matrix
            F, inliers = cv2.findFundamentalMat(mobs, kobs)

            # Essential matrix
            E = self.K[mf].transpose() @ F @ self.K[kf]
            # E, inliers = cv2.findEssentialMat(mobs, kobs, self.K[kf])

            # Pose
            points, R, t, mask = cv2.recoverPose(E, mobs[inliers, :], kobs[inliers, :])
            self.pose[kf] = np.hstack((R, t))

        else:
            K = self.K[kf]
            kobs = self.obs[kf][self.tracked[kf], :]
            trackids = self.trackids[kf][self.tracked[kf]]

            valid = np.array(tracks.valid)[trackids]
            removed = np.array(tracks.removed)[trackids]
            inliers = np.multiply(valid, ~removed).flatten()

            kobs = kobs[inliers, :]
            trackids = trackids[inliers]
            points = tracks.str[trackids, :3]

            dist = np.zeros((4, 1))
            success, w, t, inliers = cv2.solvePnPRansac(points, kobs, K, dist)
            R, _ = cv2.Rodrigues(w)
            self.pose[kf] = np.hstack((R, t))

    @staticmethod
    def pose2proj(pose):
        proj = np.vstack((pose, np.array([0, 0, 0, 1])))
        proj = np.linalg.inv(proj)
        return proj

    class graph:
        def __init__(self):
            self.conviews = []
            self.conviews.append(np.zeros(0, int))

        def update(self, kf):
            mf = kf - 1
            conviews = np.concatenate([np.array([mf], int), self.conviews[mf]])
            conviews = np.unique(conviews)
            conviews = np.sort(conviews)[::-1]
            self.conviews.append(conviews)

            for i in range(len(self.conviews[kf])):
                mf = self.conviews[kf][i]
                conviews = np.concatenate([self.conviews[mf], np.array([kf], int)])
                conviews = np.unique(conviews)
                conviews = np.sort(conviews)[::-1]
                self.conviews[mf] = conviews

    class bow:
        def __init__(self, Images, config):
            n_clusters = 10
            depth = 2
            imgs = []

            for img in Images.gray:
                img = cv2.resize(img, (int(img.shape[1] / config.sparse.sparsity),
                                       int(img.shape[0] / config.sparse.sparsity)))
                imgs.append(img)

            print('Creating Vocabulary')
            # self.vocabulary = dbow.Vocabulary(imgs, n_clusters, depth)
            # self.vocabulary.save('vocabulary.pickle')
            self.vocabulary = dbow.Vocabulary.load('vocabulary.pickle')

            detector = cv2.ORB_create()

            print('Creating Bag of Binary Words from Images')
            self.bows = []
            for image in imgs:
                kps, descs = detector.detectAndCompute(image, None)
                descs = [dbow.ORB.from_cv_descriptor(desc) for desc in descs]
                self.bows.append(self.vocabulary.descs_to_bow(descs))

            print('Creating Database')
            db = dbow.Database(self.vocabulary)
            for image in imgs:
                kps, descs = detector.detectAndCompute(image, None)
                descs = [dbow.ORB.from_cv_descriptor(desc) for desc in descs]
                db.add(descs)

            print('Querying Database')
            for image in imgs:
                kps, descs = detector.detectAndCompute(image, None)
                descs = [dbow.ORB.from_cv_descriptor(desc) for desc in descs]
                scores = db.query(descs)
                print(scores)
                match_bow = db[np.argmax(scores)]
                print(np.argsort(scores)[-5:][::-1])
                match_desc = db.descriptors[np.argmax(scores)]