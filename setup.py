from setuptools import setup

with open("README.md", "r") as fp:
    long_description = fp.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='nasnetkeras',
      version='1.0',
      description='Keras implementation of NASNet A',
      long_description=long_description,
      author='Johannes Ulén',
      url='https://github.com/johannesu/NASNet-keras/',
      license='MIT',
      install_requires=required,
      py_modules=["nasnet"],
      )
