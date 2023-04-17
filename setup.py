from setuptools import setup, find_packages

REQUIRED_PKGS = [
    "accelerate",
    "datasets>=2.8.0",
    "numpy>=1.18.2",
    "Pillow",
    "pip-chill",
    "statsmodels>=0.13.5",
    "torch>=1.4.0",
    "torchvision>=0.11.1"
    "transformers>=4.18.0",
    "tqdm",
]

setup(
    name='ailab',
    version='0.1.0',
    packages=find_packages(include=['ailab', 'ailab.*']),
    install_requires=REQUIRED_PKGS,
    extras_require={
        'interactive': ['matplotlib>=2.2.0', 'jupyter'],
    }
)