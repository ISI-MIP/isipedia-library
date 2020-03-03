"""Get details from World Bank etc
"""
import os, sys, logging

# find out the appropriate paths
for test in ['countrymasks', '../countrymasks', '../../countrymasks']:
    if os.path.exists(test):
        sys.path.append(test)
        break

try:
    import country_data
except ImportError:
    logging.warning('failed import country_data (probably countrymasks was not found)')
    raise
    
from country_data import CountryStats, CountryStatDB, countrymasks as countrymasks_folder, country_data_path as country_data_folder


class Country(CountryStats):

    @property
    def nameS(self):
        if self.name[len(self.name)-1] == 's':
          return self.name+'’'
        else:
          return self.name+'’s'   

    @property
    def thename(self):
        if self.name == 'World' or self.name in ():
          return 'the '+self.name.lower()
        else:
          return self.name

    @property
    def pop_total(self):
        return self.getvalue('POP_TOTL')

    @property
    def area(self):
        return self.getvalue('SURFACE_AREA')

