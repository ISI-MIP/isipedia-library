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
      install_requires = ['tdqm','python-frontmatter', 'jinja2'],
      extras_require = {
          'data': ['pandas', 'numpy', 'xarray', 'netCDF4'],
          'figure': ['matplotlib', 'altair'],
          'all': ['python-frontmatter', 'jinja2', 'tdqm',
                  'dimarray', 'pandas', 'numpy', 'xarray', 'netCDF4',
                  'matplotlib', 'altair'],
      },
      )

