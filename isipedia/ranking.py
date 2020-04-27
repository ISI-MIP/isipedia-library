"""Related to ranking data
"""
import json
import yaml
import os
import glob
import logging

from isipedia.jsonfile import JsonFile


class CubeVariable:
    """This class corresponds to cube definition of a variable
    """
    def __init__(self, indicator, studytype, name):
        self.indicator = indicator
        self.studytype = studytype
        self.name = name

    def jsonfile(self, area, cube_path):
        return os.path.join(cube_path, self.indicator, self.studytype, area, self.name+'_'+area+'.json')


def calculate_ranking(var, cube_path, country_names=None):
    """load ranking data"""
    if country_names is None:
        files = sorted(glob.glob(var.jsonfile('*', cube_path)))
        logging.info('found {} matching files'.format(len(files)))

        # check
        country_dirs = sorted(glob.glob(os.path.join(*os.path.split(var.jsonfile('*', cube_path))[:-1])+'/'))
        if len(country_dirs) != len(files):
            logging.warning('found {} country dirs, but {} ranking-variable files'.format(len(country_dirs), len(files)))
            areas_files = [os.path.basename(os.path.dirname(d)) for d in files]
            areas_dir = [os.path.basename(d) for d in country_dirs]
            print('set difference:', set(areas_dir).difference(set(areas_files)))

    else:
        files = [var.jsonfile(area, cube_path) for area in country_names]

    n = None
    ref = None
    data = {}
    for f in files:
        area = os.path.basename(os.path.dirname(f))
        if area == 'world':
            continue
        js = JsonFile.load(f)
        if n is None:
            n = len(js.x)
            ref = f
            data['_index'] = js.x
        else:
            assert len(js.x) == n, 'not all files have same length, found {}:{} and {}:{}'.format(ref, n, f, len(js.x))

        if hasattr(js, 'climate_scenario_list'):
            data[area] = {scenario: js.getarray(scenario) for scenario in js.climate_scenario_list}
        else:
            data[area] = {None: js.getarray()}
    # data['_plot_type'] = js.plot_type
    # data['_filename'] = var.jsonfile('world', cube_path)  # used in ranking map next to world folder...
    js = JsonFile.load(var.jsonfile('world', cube_path))  # load world-level and copy metadata
    data['_metadata'] = {k:v for k,v in js._js.items() if k != 'data'}

    return data


def load_indicator_config(indicator):
    cfgfile = os.path.join(indicator+'.yml')
    if not os.path.exists(cfgfile):
        logging.warn('no config file present for '+indicator)
    return yaml.load(open(cfgfile))


def ranking_file(indicator, category, variable, cube_path):
    # return os.path.join(cube_path, indicator, 'ranking.{}.{}.{}.json'.format(indicator, category, variable))
    return os.path.join(cube_path, indicator, category, 'world', 'ranking', 'ranking.{}.json'.format(variable))


def preprocess_ranking(indicator, cube_path, out_cube_path=None, country_names=None):
    if out_cube_path is None:
        out_cube_path = cube_path

    cfg = load_indicator_config(indicator)
    category = cfg['studytype']

    for name in cfg.get('ranking-files',[]):
        print(indicator, category, name)
        var = CubeVariable(indicator, category, name)
        data = calculate_ranking(var, cube_path, country_names=country_names)
        fname = ranking_file(indicator, category, name, out_cube_path)
        dname = os.path.dirname(fname)
        if not os.path.exists(dname):
            os.makedirs(dname)
        json.dump(data, open(fname, 'w'))



def ordinal(num):
    if 10 <= num % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1:'st', 2:'nd', 3:'rd'}.get(num % 10, 'th')
    return '{}<sup>{}</sup>'.format(num, suffix)


class Ranking:
    def __init__(self, data):
        self.data = data
        self.areas = sorted([area for area in data if not area.startswith('_')])
        # self.filename = data['_filename']  # for the figure...

    def __getattr__(self, name):
        return self.data['_metadata'][name] # e.g. filename, plot_type (same as JsonFile)

    def value(self, area, x, scenario=None):
        index = self.data['_index'].index(x)
        if self.plot_type == 'indicator_vs_temperature':
            return self.data[area]['null'][index]
        else:
            return self.data[area][scenario][index]

    def values(self, x, scenario=None):
        index = self.data['_index'].index(x)
        if self.plot_type == 'indicator_vs_temperature':
            values = [self.data[area]['null'][index] for area in self.areas]
        else:
            values = [self.data[area][scenario][index] for area in self.areas]
        return values

    def sorted_areas(self, x, scenario=None):
        values = self.values(x, scenario)
        pairs = sorted([(v if v is not None else -99e9, area) for v, area in zip(values, self.areas)], key=lambda x: x[0])
        return [area for v, area in pairs[::-1]]

    def number(self, area, x, scenario=None):
        if self.value(area, x, scenario) is None:
            return 'undefined'
        return self.sorted_areas(x, scenario).index(area) + 1

    def position(self, area, x, scenario=None):
        if self.value(area, x, scenario) is None:
            return 'undefined'
        num = self.number(area, x, scenario)
        return ordinal(num)


    @classmethod
    def load(cls, fname):
        ranking_data = json.load(open(fname))
        return cls(ranking_data)
