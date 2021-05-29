#!/usr/bin/env bash
OUT="$PWD"
cd /tmp/
wget https://gitlab.com/api/v4/projects/4207231/packages/generic/graphviz-releases/2.47.2/graphviz-2.47.2.tar.gz
tar xf graphviz-2.47.2.tar.gz
cd graphviz-2.47.2
./configure --prefix=$OUT/graphviz.ppc64le --disable-perl --disable-java --disable-lua --enable-python3
make -j8
make install
