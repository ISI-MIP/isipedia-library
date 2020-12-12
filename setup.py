from setuptools import setup
import versioneer

setup(name='isipedia',
      version=versioneer.get_version(),
      cmdclass = versioneer.get_cmdclass(),
      author='Mahe Perrette for ISIpedia',
      author_email='mahe.perrette@pik-potsdam.de',
      description='utilities to crunch data for ISIpedia',
      url='git@github.com:ISI-MIP/isipedia-library.git',
      packages=['isipedia'],
      scripts=['scripts/isipedia_build.py'],
      license = "MIT",
      install_requires=open('requirements.txt').read().splitlines(),
      )

