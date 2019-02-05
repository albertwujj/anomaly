import random
import math
import numpy as np
import numpy.ma as ma

random.seed(1957)
np.random.seed(1958)

s = 1000
m = 8
p = 1000
w = 4

def example_hash(l):
    ret = 0
    for i, x in enumerate(l):
        ret += x * 2**i
    return int(ret) % p

data = np.array([line.split() for line in open('data.txt').readlines()],dtype=float)
dims = data.shape[1]
hashes = [example_hash] * 4

fs = [random.uniform(1 / math.sqrt(s), 1 - 1 / math.sqrt(s)) for _ in range(m)]
ls = [math.log(s, max(1/f, 2)) for f in fs]
rs = [round(random.uniform(1 + .5 * l,l)) for l in ls]
chosen_dims = [np.random.choice(dims, size=min(dims,r), replace=False) for r in rs]

ensemble_data = [np.take(np.take(data, np.random.choice(data.shape[0], size=s, replace=True), axis=0),
                 dims, axis=1) for dims in chosen_dims] # take random dimensions from random samples

emins = [e.min(0) for e in ensemble_data]
emaxs = [e.max(0) for e in ensemble_data]
normalized_data = [(e - emins[i]) / (emins[i] + emaxs[i]) for i, e in enumerate(ensemble_data)]
os = [np.array([random.uniform(0, fs[i]) for _ in range(e.shape[1])]) for i, e in enumerate(ensemble_data)]
offset_data = [(np.expand_dims(os[i], 0) + ne) / fs[i] for i, ne in enumerate(normalized_data)]

hash_poses_matrix = np.asarray([[np.apply_along_axis(hfunc, 1, e) for hfunc in hashes] for e in offset_data])
def count_index(hash_poses):
    ret = np.zeros((p))
    np.add.at(ret, hash_poses, 1)
    return ret
minsketch = np.apply_along_axis(count_index, -1, hash_poses_matrix)

print("fs, rs: {} {}".format(np.average(np.array(fs)), np.average(np.array(rs))))
print(np.max(minsketch))
mslist = minsketch.tolist()
#[print(w) for w in minsketch[0]]

normal = [random.uniform(-1000, 1000) for _ in range(100)]
outlier = [random.uniform(-200, 1000) for _ in range(100)]
test_data = np.asarray([normal, outlier])

test_data = [np.take(test_data, dims, axis=1) for dims in chosen_dims]
test_data = [(te - emins[i]) / (emins[i]) + (emaxs[i]) for i, te in enumerate(test_data)]
test_data = [(np.expand_dims(os[i], 0) + te) / fs[i] for i, te in enumerate(test_data)]
test_data_hashed = np.asarray([[np.apply_along_axis(hfunc, 1, te) for hfunc in hashes] for te in test_data])

scores = np.take_along_axis(minsketch, test_data_hashed, -1)
scores = scores.min(axis=1)
scores = np.log2(scores + 1)
scores = np.average(scores, axis=0)
print(scores)
