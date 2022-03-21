#%% Load data
import pickle

import matplotlib.scale
import numpy as np

LOCAL = "/code/compy-learn/local"
from compy.representations.sequence_graph import Vocabulary, SequenceGraph

with open("/code/compy-learn/local/cvevulns-tokens.pickle", 'rb') as f:
    data = pickle.load(f)
#%% Pandas
import pandas as pd
df = pd.json_normalize(data['samples'], sep='_')
#%% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%% Sequence lengths
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

reps = [s['x']['code_rep'] for s in data['samples']]
stats = pd.DataFrame({
    'seq_len': [r.seq_len for r in reps],
    'node_count': [len(r.nodes) for r in reps],
})
stats.seq_len.clip(upper=1500).hist()
plt.title("Histogram of sequence lengths")
plt.show()
#%% Node counts
import scipy.stats
estimated = scipy.stats.pareto(0.341970645124522, -0.03476278156661178, 16.034762627198077)
pd.DataFrame({
    'node_count': stats.node_count.clip(upper=4000),
    'estimated': pd.Series(estimated.rvs(stats.node_count.size)).clip(upper=4000)
}).hist(bins=40)

plt.show()
#%% Correlation length/vuln
vuln_function_names = set(df[df['y'].astype(np.bool)].info_name)
has_vuln_pair = df.info_name.isin(vuln_function_names)
seq_len = df.x_code_rep.apply(lambda r: r.seq_len).clip(upper=10000)
mask = seq_len > 1300
pd.DataFrame({
    'vuln_pair': seq_len[has_vuln_pair & mask],
    'no_vuln_pair': seq_len[~has_vuln_pair & mask],
}).hist(sharey=True)
plt.show()
