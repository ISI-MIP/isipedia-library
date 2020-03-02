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
    
from country_data import CountryStats, CountryStatDB, countrymasks as countrymasks_folder