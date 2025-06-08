from setuptools import setup, find_packages

setup(
    name='predictris',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'coclust',
        'matplotlib',
        'networkx',
        'numpy',
        'pyvis',
        'scikit-learn',
        'scipy',
        'seaborn',
        'tqdm',
    ],
)