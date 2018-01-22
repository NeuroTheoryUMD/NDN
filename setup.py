from setuptools import setup

with open('README.md') as f:
    readme = f.read()

requirements = [
    'numpy>=1.12.1',
    'scipy',
    'tensorflow',
]

setup(
    # Metadata
    name='ndn',
    version=2.0,
    author='Neurotheory Lab',
    author_email='dab@umd.edu',
    url='https://github.com/NeurotheoryUMD/NDN',
    description='framework for modeling neural data',
    long_description=readme,
    license='MIT',
    install_requires=requirements,
)

