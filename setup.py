"""Setup script for object_detection with TF2.0."""
import os
from setuptools import find_packages
from setuptools import setup

# Note: adding apache-beam to required packages causes conflict with
# tf-models-offical requirements. These packages request for incompatible
# oauth2client package.
REQUIRED_PACKAGES = [
    # Required for apache-beam with PY3
    'absl-py>=0.8.1',
    'astor>=0.8.0',
    'Click>=7.0',
    'gast>=0.3.2',
    'google-pasta>=0.1.7',
    'grpcio>=1.24.1',
    'h5py>=2.10.0',
    'itsdangerous>=1.1.0',
    'Jinja2>=2.10.3',
    'Keras-Applications>=1.0.8',
    'Keras-Preprocessing>=1.1.0',
    'Markdown>=3.1.1',
    'MarkupSafe>=1.1.1',
    'numpy>=1.17.2',
    'opencv-python',
    'Pillow>=6.2.0',
    'protobuf>=3.10.0',
    'six>=1.12.0',
    'tensorboard==2.5.0',
    'tensorflow==2.5.0',
    'tensorflow-estimator==2.5.0',
    'termcolor==1.1.0',
    'Werkzeug==0.16.0',
    'wincertstore==0.2',
    'wrapt>=1.11.2',
    'tf_slim',
    'gdown',
    'Cython',
    'PyYAML>=5.3',
    'tqdm>=4.41.0',
    'scipy>=1.4.1',
    'torch==1.8.2+cpu',
    'torchvision==0.9.2+cpu',
    'fvcore',
    'cloudpickle',
    'pandas',
    'seaborn',
    'torch==1.8.2+cpu',
    'torchvision==0.9.2+cpu',
]

setup(
    name='alphadetector',
    version='0.0.8',
    author='ravikanur',
    url='https://github.com/ravikanur/ObjectDetection_Webapp',
    description='object detection using yolo, tf2 and detectron2',
    project_urls={
        'Bug Tracker': 'https://github.com/ravikanur/ObjectDetection_Webapp/issues',
    },
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    dependency_links=['https://download.pytorch.org/whl/lts/1.8/torch_lts.html'],
    package_dir={"":'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.7',
)
