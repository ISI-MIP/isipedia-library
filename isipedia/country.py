import json
import os
import warnings
import logging
import glob

try:
    from country_data import country_data_folder, countrymasks_folder
except ImportError:
    warnings.warn("See https://github.com/ISI-MIP/isipedia-countries for install instruction")
    raise


class Country:
    """This is the class for the corresponding json file in country_data
    """

    def __init__(self, name, type="country", sub_countries=[], code=None, stats=None):
        self.name = name
        self.type = type
        self.code = code
        self.sub_countries = sub_countries
        self.stats = stats or []

    def get(self, name, insert=False):
        try:
            i = [e['type'] for e in self.stats].index(name)
            return self.stats[i]
        except ValueError:
            if insert:
                e = {'type': name}
                self.stats.append(e)
                return e
            else:
                raise

    def getvalue(self, name, missing=float('nan')):
        try:
            return self.get(name)['value']
        except ValueError:
            return missing

    @classmethod
    def load(cls, fname):
        js = json.load(open(fname))
        code = os.path.basename(os.path.dirname(fname))
        return cls(js['name'], js.get('type', 'country'), js.get('sub-countries',[]), code=js.get('code', code), stats=js.get('stats', []))

    def save(self, fname):
        cdir = os.path.dirname(fname)
        if not os.path.exists(cdir):
            logging.info('create '+repr(cdir))
            os.makedirs(cdir)

        js = {
            'name': self.name,
            'code': self.code,
            'type': self.type,
            'sub-countries': self.sub_countries,
            'stats': self.stats,
        }
        json.dump(js, open(fname, 'w'))


    def __repr__(self):
        return 'Country({name}, {code})'.format(**vars(self))

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



def get_country(code):
    """return Country class
    """
    i = [c.lower() for c in countries_codes].index(code.lower())
    return countries[i]


# read all countries on import
countries = []
countries_codes = []
countries_names = []

# sort by code name, world first
_sort_key = lambda x: os.path.basename(os.path.dirname(x)).lower().replace('world', '_')

for jsonpath in sorted(glob.glob(os.path.join(country_data_folder, '*', '*_general.json')), key=_sort_key):
    country = Country.load(jsonpath)
    countries.append(country)
    countries_codes.append(country.code)
    countries_names.append(country.name)