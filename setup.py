# Inspiration: https://stackoverflow.com/a/24188094/793218
#

from setuptools import setup, find_packages, Extension
import subprocess

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

subprocess.call(['make', '-C', 'src'])

setup(
    name='mickey',
    version='0.2',
    description='Assorted methods and classes to handle and visualize the output of the Pluto MHD code',
    long_description=readme,
    author='Rodrigo Nemmen',
    author_email='rodrigo.nemmen@iag.usp.br',
    url='https://bitbucket.org/nemmen/mickey',
    license=license,
    #ext_modules=[Extension('_fastregrid', ['fastregrid.c'])],
    packages=find_packages(exclude=('tests', 'docs')),
    package_data={'fastregrid': ['_fastregrid.so']}
)