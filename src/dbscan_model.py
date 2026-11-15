import numpy as np

class CustomDBSCAN:
    def __init__(self,eps,min_samples,metrics = "haversine"):
        self.eps = eps
        self.min_samples = min_samples
        self.metrics = metrics
        self.labels_ = None
        self.core_sample_indices_ = None
        self.components_ = None
    def _get_neighbors(self,X,point_idx):
        if self.metrics == "haversine":
            lat = X[:,0]
            long = X[:,1]
            lat_p,long_p = X[point_idx,0],X[point_idx,1]
            dlat = lat-lat_p
            dlong = long-long_p
            a = np.sin(dlat/2)**2 + np.cos(lat_p)*np.cos(lat)*np.sin(dlong/2)**2
            a = np.clip(a,0.0,1.0)
            dists = 2.0*np.arcsin(np.sqrt(a))
        else:
            diff = X-X[point_idx]
            dists = np.sqrt(np.sum(diff**2,1))
        return np.where(dists <= self.eps)[0].tolist()

    def fit(self,X):
        n_samples = len(X)
        self.labels_ = np.full(n_samples,-2,dtype = np.int32)
        cluster_id = 0
        core_indices = []
        for i in range(n_samples):
            if self.labels_[i] != -2:
                continue
            neighbors = self._get_neighbors(X,i)
            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1
                continue
            core_indices.append(i)
            self.labels_[i] = cluster_id
            queue = [idx for idx in neighbors if idx != i]
            in_queue = set(queue)
            while queue:
                current = queue.pop(0)
                if self.labels_[current] == -1:
                    self.labels_[current] = cluster_id
                if self.labels_[current] != -2:
                    continue
                self.labels_[current] = cluster_id
                sub_neighbors = self._get_neighbors(X,current)
                if len(sub_neighbors) >= self.min_samples:
                    core_indices.append(current)
                    for sn in sub_neighbors:
                        if sn not in in_queue and self.labels_[sn] in [-2,-1]:
                            queue.append(sn)
                            in_queue.add(sn)
            cluster_id += 1
        self.core_sample_indices_ = np.array(list(set(core_indices)),dtype = np.int32)
        self.components_ = X[self.core_sample_indices_] \
                if len(self.core_sample_indices_) > 0 else np.empty((0,X.shape[1]))
        return self
    def fit_predict(self,X):
        self.fit(X)
        return self.labels_
    def predict(self,X_new):
        if self.components_ is None or len(self.components_) == 0:
            return np.full(len(X_new),-1,dtype = np.int32)
        n_new = len(X_new)
        pred_labels = np.full(n_new,-1,dtype = np.int32)
        core_labels = self.labels_[self.core_sample_indices_]
        for i in range(n_new):
            p = X_new[i]
            if self.metrics == "haversine":
                dlat = self.components_[:,0] - p[0]
                dlong = self.components_[:,1] - p[1]
                a = (np.sin(dlat / 2.0) ** 2 + np.cos(p[0]) * np.cos(self.components_[:, 0])
                     * np.sin(dlong / 2.0) ** 2)
                a = np.clip(a, 0.0, 1.0)
                dists = 2.0*np.arcsin(np.sqrt(a))
            else:
                diff = self.components_ - p
                dists = np.sqrt(np.sum(diff**2,1))
            min_idx = np.argmin(dists)
            if dists[min_idx] <= self.eps:
                pred_labels[i] = core_labels[min_idx]
        return pred_labels
if __name__ == "__main__":
    customDB = CustomDBSCAN(1,3)
    X = np.stack([np.random.uniform(1,10,13),np.random.uniform(1,10,13)],axis=1)
    a = customDB._get_neighbors(X,5)

