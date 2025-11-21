from setuptools import setup, find_packages

setup(
    name='yolov11-people-detection',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A project for detecting people using YOLOv11 model.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch>=1.7.0',
        'torchvision>=0.8.0',
        'opencv-python',
        'numpy',
        'matplotlib',
        'pandas',
        'scikit-learn',
        'pyyaml',
        'tqdm',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)