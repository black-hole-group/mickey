from setuptools import setup, find_packages, Extension

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

_fastregrid = Extension("_fastregrid",
                   ["fastregrid.i","fastregrid.c"],
                   #include_dirs = [numpy_include],
                   )

setup(
    name='mickey',
    version='0.2',
    description='Assorted methods and classes to handle and visualize the output of the Pluto MHD code',
    long_description=readme,
    author='Rodrigo Nemmen',
    author_email='rodrigo.nemmen@iag.usp.br',
    url='https://bitbucket.org/nemmen/mickey',
    license=license,
    ext_modules=[Extension('_fastregrid', ['fastregrid.c'])],
    packages=find_packages(exclude=('tests', 'docs'))
)