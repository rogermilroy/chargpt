from setuptools import setup, find_packages

setup(
    name='balanced_sampler',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'torch>=1.6.0',
        'torchvision>=0.2.1',
        'tqdm>=4.31.1'
    ],
)
