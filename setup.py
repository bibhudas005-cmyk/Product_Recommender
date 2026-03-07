from setuptools import setup,find_packages


with open("requirements.txt","r") as f:
    requirements=f.read().splitlines()

setup(
    name="Flipkart Recommender",
    version="0.1",
    author="Ayush",
    packages=find_packages(),
    install_requires=requirements,

)



#To trigger this setup.py we will be using this cmd-: pip install -e .