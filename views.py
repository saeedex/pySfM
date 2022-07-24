from config import *

class views:
    def __init__(self, Images, config):
        self.sparse = views.sparse()
        self.dense = []
        self.graph = views.graph()
        self.camera = views.camera(Images, config)

    class sparse:
        def __init__(self):
            self.dsc = []
            self.obs = []
            self.matched = []
            self.tracked = []
            self.tracks = []

        # detect features in loaded images
        def detect(self, Images, config):
            for img in Images.color:
                kpts, dscs = config.sparse.detector.detectAndCompute(img, None)
                obs = np.float32([kpts[i].pt for i in range(len(kpts))])

                self.dsc.append(dscs)
                self.obs.append(obs)
                self.matched.append(np.zeros(obs.shape[0], dtype=bool))
                self.tracked.append(np.zeros(obs.shape[0], dtype=bool))
                self.tracks.append(np.zeros(obs.shape[0], dtype=int))

        def addtrack(self, Track, tracks, kf, kidx):
            self.matched[kf][kidx] = True
            self.tracked[kf][kidx] = Track.valid[tracks]
            self.tracks[kf][kidx] = tracks

        def rmvtrack(self, Track, tracks):
            for j in range(len(tracks)):
                i = tracks[j]

                for k in range(len(Track.views[i])):
                    kf = Track.views[i][k]
                    idx = Track.idx[i][k]
                    self.tracks[kf][idx] = 0
                    self.tracked[kf][idx] = False
                    self.matched[kf][idx] = False

        def mrgtrack(self, Track, tracks, dtracks):
            for j in range(len(tracks)):
                i = tracks[j]
                for k in range(len(Track.views[dtracks[j]])):
                    kf = Track.views[dtracks[j]][k]
                    kidx = Track.idx[dtracks[j]][k]
                    self.tracks[kf][kidx] = i

        def acttrack(self, Track, tracks):
            for j in range(len(tracks)):
                i = tracks[j]

                for k in range(len(Track.views[i])):
                    kf = Track.views[i][k]
                    idx = Track.idx[i][k]
                    self.tracked[kf][idx] = Track.valid[i]

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
            #self.vocabulary = dbow.Vocabulary(imgs, n_clusters, depth)
            #self.vocabulary.save('vocabulary.pickle')
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

    class camera:
        def __init__(self, Images, config):
            self.pose = []
            self.K = []

            # init poses and intrinsics
            opose = np.hstack((np.eye(3, 3), np.zeros((3, 1))))

            f = np.max(Images.gray[0].shape)*config.dataset.intscale
            params = [f, f, Images.gray[0].shape[1]/2, Images.gray[0].shape[0]/2]
            K = np.array([[params[0], 0, params[2]], [0, params[1], params[3]], [0, 0, 1]])

            for kf in range(len(Images.color)):
                self.pose.append(opose)
                self.K.append(K)


        def loadintr(self, Images, config):
            datafile = open(config.dataset.path + "/" + 'cameraIntrinsic.txt')
            params = list()

            for line in datafile:
                strs = line.split(' ', line.count(' '))
                for strvar in strs:
                    params.append(float(strvar))

            if Images.color[1].shape[0] > 640:
                params = params * 3

            K = np.array([[params[0], 0, params[2]], [0, params[1], params[3]], [0, 0, 1]])
            K[:2, :] = K[:2, :] / config.dataset.imresize

            for kf in range(len(Images.color)):
                self.K[kf] = K

        def loadposes(self, Images, config):
            datafile = open(config.dataset.path + "/" + 'cameras.out')
            data = list()

            for line in datafile:
                linedata = list()
                strs = line.split(' ', line.count(' '))
                for strvar in strs:
                    linedata.append(float(strvar))
                data.append(linedata)

            for kf in range(len(Images.color)):
                ext = np.array(data[kf * 5 + 1:kf * 5 + 5])
                R = ext[:3, :]
                t = np.reshape(ext[3, :], (3, 1))

                pose = np.hstack((R, t))
                pose[1:, :] = -pose[1:, :]
                self.pose[kf] = pose

        def computepose(self, Views, Track, kf):
            if kf == 1:
                mf = kf - 1
                tracks = Views.sparse.tracks[kf][Views.sparse.matched[kf]]
                fidx = np.array(Track.idx)[tracks]
                midx = fidx[:, 0]
                kidx = fidx[:, 1]

                mobs = Views.sparse.obs[mf][midx, :]
                kobs = Views.sparse.obs[kf][kidx, :]

                # Fundamental matrix
                F, inliers = cv2.findFundamentalMat(mobs, kobs)

                # Essential matrix
                E = self.K[mf].transpose() @ F @ self.K[kf]
                #E, inliers = cv2.findEssentialMat(mobs, kobs, self.K[kf])

                # Pose
                points, R, t, mask = cv2.recoverPose(E, mobs[inliers, :], kobs[inliers, :])
                self.pose[kf] = np.hstack((R, t))

            else:
                K = self.K[kf]
                kobs = Views.sparse.obs[kf][Views.sparse.tracked[kf], :]
                tracks = Views.sparse.tracks[kf][Views.sparse.tracked[kf]]

                valid = np.array(Track.valid)[tracks]
                removed = np.array(Track.removed)[tracks]
                inliers = np.multiply(valid, ~removed).flatten()

                kobs = kobs[inliers, :]
                tracks = tracks[inliers]
                points = Track.str[tracks, :3]

                dist = np.zeros((4, 1))
                success, w, t, inliers = cv2.solvePnPRansac(points, kobs, K, dist)
                R, _ = cv2.Rodrigues(w)
                self.pose[kf] = np.hstack((R, t))

        @staticmethod
        def pose2proj(pose):
            proj = np.vstack((pose, np.array([0, 0, 0, 1])))
            proj = np.linalg.inv(proj)
            return proj