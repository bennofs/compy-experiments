#!/usr/bin/env python3
import collections
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import json
import gzip
import glob
import pickle
#%%
reveal_data = pd.read_csv("/mnt/ml4code/reveal/debian_data.csv")
#%%
vn = pd.read_json("/mnt/ml4code/reveal/vulnerables.json")
nv = pd.read_json("/mnt/ml4code/reveal/non-vulnerables.json")
deb_data = reveal_data[reveal_data.project == 'debian']

vn_hashes = set(vn.hash)
nv_hashes = set(nv.hash)
#%%
def load_cve(fname):
    with gzip.open(fname) as f:
        return json.load(f)['CVE_Items']


def load_all_cves(dirname):
    return [cve for fname in glob.glob(dirname + "/nvdcve-*.json.gz") for cve in load_cve(fname)]

all_cves = load_all_cves("/mnt/ml4code/nvd")
#%%
text2cve = { cve['cve']['description']['description_data'][0]['value']: cve['cve']['CVE_data_meta']['ID'] for cve in all_cves }
relevant_data = deb_data[deb_data.hash.isin(vn_hashes | nv_hashes)]
cve_files = relevant_data.assign(cve=relevant_data.text.map(lambda t: text2cve.get(t)))

manual_cves = {
    'use after free resulting from failure to skip invalid packets': 'CVE-2015-1606',
    'CVE (at NVD; CERT, LWN, oss-sec, fulldisc, bugtraq, EDB, Metasploit, Red Hat, Ubuntu, Gentoo, SUSE bugzilla/CVE, Mageia, GitHub code/issues, web search, more)': 'CVE-2015-7851',
    'The VC-2 Video Compression encoder in FFmpeg 3.4 allows remote attackers to cause a denial of service (out-of-bounds read) because of incorrect buffer padding for non-Haar wavelets, related to libavcodec/vc2enc.c and libavcodec/vc2enc_dwt.c.': 'CVE-2017-16840',
    'data-dependent timing variations in modular exponentiation': 'CVE-2015-0837',
    'sidechannel attack on Elgamal': 'CVE-2014-3591',
    'Invalid memory access in rtp code': 'CVE-2014-9630',
    "In libavformat/nsvdec.c in FFmpeg 3.3.3, a DoS in nsv_parse_NSVf_header() due to lack of an EOF (End of File) check might cause huge CPU consumption. When a crafted NSV file, which claims a large \"table_entries_used\" field in the header but does not contain sufficient backing data, is provided, the loop over 'table_entries_used' would consume huge CPU resources, since there is no EOF check inside the loop.": 'CVE-2017-14171',
    'buffer overflow in virtio-serial': 'CVE-2015-5745',
    'integer overflow with resultant buffer overflow': 'CVE-2014-9629',
    'getaddrinfo(), glob_in_dir stack overflow': 'CVE-2013-4357',
    'A specially constructed response from a malicious server can cause a buffer overflow in dhclient': 'CVE-2018-5732',
}

extra_cves = cve_files[cve_files.cve.isna()].text.map(lambda x: manual_cves[x])
cve_files = cve_files.assign(cve=cve_files.cve.fillna(extra_cves))

cve_files.at[1890, 'cve'] = 'CVE-2014-4609'
cve_files.at[1891, 'cve'] = 'CVE-2014-4609'

cve_files.info()
#%%
from compy.datasets.cvevulns import CVEVulnsDataset
dataset = CVEVulnsDataset()

dataset_md = pd.read_json(Path(dataset.content_dir) / "metadata.json", orient='index')
functions = pd.read_json(Path(dataset.content_dir) / 'functions.json')
#%%
with open("/code/compy-learn/local/cvevulns-tokens.pickle", 'rb') as f:
    data = pickle.load(f)

with open("/code/compy-experiments/local/cvevulns-tokens-paired.pickle", 'rb') as f:
    data_paired = pickle.load(f)
#%%
used_files_count = len(set(s['info']['file_idx'] for s in data['samples']))
compiled_files_ratio = used_files_count / len(cve_files)
print(f"used {used_files_count} of len {len(cve_files)}, {dataset_md.pp_name.count()} preprocssed files, compilation ratio {compiled_files_ratio*100:02.1f}%")
#%% Functions by CVE
cve_count = pd.Series(collections.Counter(s['info']['cve'] for s in data['samples'] if s['y'])).sort_values(ascending=False)
cve_count_paired = pd.Series(collections.Counter(s['info']['cve'] for s in data_paired['samples'] if s['y'])).sort_values(ascending=False)

patches, _ = plt.pie(cve_count)
plt.legend(patches[:10], cve_count.index[:10], loc="upper left", bbox_to_anchor=(1,1))
plt.savefig("presentation/media/reveal-unbalanced.pdf")
