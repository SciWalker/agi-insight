# agi-insight

This is a repo consists of two components:
1. openai-handler: a python package that wraps OpenAI API
2. langchain: a python package that wraps the langchain and OpenAI API and use it to analyze a repo


Troubleshootings:
if there is issue with CUDA library path, try (only applicable to linux or wsl):
pip3 uninstall -y bitsandbytes 
git clone https://github.com/TimDettmers/bitsandbytes.git
cd bitsandbytes || exit
make CUDA_VERSION=112  # or 117, depending on the local CUDA version 
python setup.py install 