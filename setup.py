from setuptools import find_packages
from setuptools import setup

with open("./VERSION.txt", "r") as fp:
    __VERSION__ = fp.readline().strip()

setup(
    name='alpha2',
    author='Feng Xu',
    packages=[package for package in find_packages() if package.startswith("alpha2")], 
    python_requires='>=3.8',
    package_data={
        # include default config files and env data files
        "": ["*.yaml", "*.xml", "*.json"],
    }, 
    version=__VERSION__, 
    url="https://github.com/x35f/alpha2", 
)
