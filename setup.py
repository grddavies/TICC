from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name="ticclib",
    version="0.1",
    packages=find_packages(exclude=('tests', 'src', 'paper code')),
    description=("Python implementation of TICC method for segmenting and"
                 " clustering multivariate time series."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/hedscan/TICC',
    license=license,
      )
