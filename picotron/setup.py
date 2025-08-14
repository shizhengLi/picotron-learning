from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt') as req:
        return [line.strip() for line in req if line.strip() and not line.startswith('#')]

setup(
    name="picotron",
    version='0.1.0',
    packages=find_packages(),
    install_requires=read_requirements(),
)