# install setuptools (provides distutils for Python 3.12+)
pip install setuptools

# install numpy first (to avoid torch pulling in numpy 2.x)
pip install numpy==1.26.4

# install torch 2.7.1 (required by xformers 0.0.31)
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -U xformers==0.0.31 --index-url https://download.pytorch.org/whl/cu128

# reinstall numpy to ensure correct version (torch may have upgraded it)
pip install numpy==1.26.4 --force-reinstall

# install dependencies
pip install -r requirements.txt

# install packages that require torch at build time
pip install --no-build-isolation git+https://github.com/ashawkey/diff-gaussian-rasterization/
pip install --no-build-isolation "nvdiffrast@git+https://github.com/ShenhanQian/nvdiffrast@backface-culling"
pip install --no-build-isolation git+https://github.com/camenduru/simple-knn/
pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git

# === If you fail to install some modules due to network connection, you can also try the following: ===
# git clone https://github.com/facebookresearch/pytorch3d.git
# pip install --no-build-isolation ./pytorch3d
# git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
# pip install ./diff-gaussian-rasterization
# git clone https://github.com/camenduru/simple-knn.git
# pip install ./simple-knn

cd external/landmark_detection/FaceBoxesV2/utils/
sh make.sh
cd ../../../../
