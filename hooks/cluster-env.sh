WORKDIR=$(mktemp -d)
export WORKDIR

# setup the ml stack
module load modenv/ml CMake/3.15.3-GCCcore-8.3.0 PyTorch TensorFlow scikit-learn PyTorch-Geometric

# bring libs into path
export PATH="$LLVM_ROOT/bin:$PATH"
export VENV="$WORKDIR/venv"

# create a new virtualenv
python -m venv --system-site-packages "$VENV"
source "$VENV/bin/activate"

# install basic environment
pip install -U pip==21.1.2
env LDFLAGS="-L$GRAPHVIZ_ROOT/lib -Wl,-rpath,$GRAPHVIZ_ROOT/lib" \
  CPPFLAGS="-I$GRAPHVIZ_ROOT/include" \
  pip install -r "$EXPERIMENTS_ROOT/requirements.txt"