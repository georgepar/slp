from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('README.md') as f:
    README = f.read()

setup(name='slp',
      version='0.0.1',
      description='Speech and Language processing utilities '
                  'for sklearn and pytorch.',
      url='https://github.com/georgepar/slp',
      author='Giorgos Paraskevopoulos',
      author_email='georgepar.91@gmail.com',
      license='MIT',
      packages=find_packages(exclude=['docs', 'tests*', 'tools']),
      install_requires=requirements,
      include_package_data=True,
      long_description=README
      )
