from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='mickey',
    version='0.1',
    description='Assorted methods and classes to handle and visualize the output of the Pluto MHD code',
    long_description=readme,
    author='Rodrigo Nemmen',
    author_email='rodrigo.nemmen@iag.usp.br',
    url='https://bitbucket.org/nemmen/mickey',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)