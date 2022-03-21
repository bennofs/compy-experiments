#%% Load data
import pickle

import matplotlib.scale
import numpy as np
from typing import List

from compy.representations import Graph

LOCAL = "/code/compy-learn/local"
from compy.representations.sequence_graph import Vocabulary, SequenceGraph

with open("/code/compy-learn/local/cvevulns-tokens.pickle", 'rb') as f:
    data = pickle.load(f)
#%%
import collections
vuln_samples_per_cve = collections.Counter(s['info']['cve'] for s in data['samples'] if s['y'])

# discard cves that have more than 10 vulnerable functions or which are too long
samples_small = [s for s in data['samples'] if vuln_samples_per_cve[s['info']['cve']] <= 10 and s['x']['code_rep'].seq_len <= 1500]

rem_vuln_count = sum(1 for s in samples_small if s['y'])
print(f"remaining vulnerable samples: {rem_vuln_count}")
#%% Flatten the graphs of samples
def flatten_graph(graph: Graph):
    flat = graph.map_to_leaves().without_self_edges()
    return flat
#%% Make sample graphs undirected
def to_undirected(samples):
    out = []
    for sample in samples:
        new_code_rep = sample['x']['code_rep'].to_undirected()
        out.append(dict(sample, x=dict(sample['x'], code_rep=new_code_rep)))
    return out
#%% Produce "paired" (vuln and not-vuln) dataset
samples_positive = [sample for sample in samples_small if sample['y']]
vuln_funcs = set((s['info']['cve'], s['info']['name']) for s in samples_positive)
samples_hasvuln = [s for s in samples_small if (s['info']['cve'], s['info']['name']) in vuln_funcs]

ratio_vuln = sum(1 for s in samples_hasvuln if s['y']) / len(samples_hasvuln)
print(f"paired samples: {len(samples_hasvuln)} ({ratio_vuln*100:02.1f}% vulnerable)")

with open("local/cvevulns-tokens-paired.pickle", 'wb') as f:
    pickle.dump(dict(data, samples=samples_hasvuln), f)

with open("local/cvevulns-tokens-paired-undirected.pickle", 'wb') as f:
    pickle.dump(dict(data, samples=to_undirected(samples_hasvuln)), f)
#%% Produce "exclusive" (vuln XOR not vuln and other samples for balance) dataset
vuln_names = set(s['info']['name'] for s in samples_small if s['y'])

rng = np.random.default_rng(seed=42)
samples_exclusive = []
for name in vuln_names:
    candidates = [dict(s, y=1, orig_y=s['y']) for s in samples_small if s['info']['name'] == name]
    samples_exclusive.append(rng.choice(candidates))

ratio_vuln = sum(1 for s in samples_exclusive if s['orig_y']) / len(samples_exclusive)
print(f"picked {len(samples_exclusive)} functions modified in patches, {ratio_vuln*100:02.1f} percent are the vulnerable version")

candidates_never_vuln = [dict(s, orig_y=s['y']) for s in samples_small if s['info']['name'] not in vuln_names]
rng.shuffle(candidates_never_vuln)
samples_exclusive += candidates_never_vuln[:len(samples_exclusive)]
ratio_had_vuln = sum(1 for s in samples_exclusive if s['y']) / len(samples_exclusive)
ratio_is_vuln = sum(1 for s in samples_exclusive if s['orig_y']) / len(samples_exclusive)
print(f"final number of samples: {len(samples_exclusive)}, {ratio_had_vuln*100:02.1f}% have a vuln version, {ratio_is_vuln*100:02.1f}% are vuln")
with open("local/cvevulns-tokens-exclusive.pickle", 'wb') as f:
    pickle.dump(dict(data, samples=samples_exclusive), f)
with open("local/cvevulns-tokens-exclusive-undirected.pickle", 'wb') as f:
    pickle.dump(dict(data, samples=to_undirected(samples_exclusive)), f)
