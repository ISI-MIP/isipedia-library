"""Related to ranking data
"""
import json
import os
import glob
import logging
from tqdm import tqdm

from isipedia.jsonfile import JsonFile, CsvFile

def calculate_ranking(study_path, name, country_names=None):
    """load ranking data"""
    csvfile = lambda area: os.path.join(study_path, f'{area.lower()}', name+f'_{area}.csv')

    if country_names is None:
        files = sorted(glob.glob(csvfile(area='*')))
        # logging.info('found {} matching files'.format(len(files)))

        ## check
        #country_dirs = sorted(glob.glob(os.path.join(*os.path.split(var.csvfile('*', cube_path))[:-1])+'/'))
        #if len(country_dirs) != len(files):
        #    logging.warning('found {} country dirs, but {} ranking-variable files'.format(len(country_dirs), len(files)))
        #    areas_files = [os.path.basename(os.path.dirname(d)) for d in files]
        #    areas_dir = [os.path.basename(d) for d in country_dirs]
        #    print('set difference:', set(areas_dir).difference(set(areas_files)))

    else:
        files = [csvfile(area=area) for area in country_names]

    n = None
    ref = None
    data = {}

    for f in tqdm(files):
        area = os.path.basename(os.path.dirname(f))
        if area == 'world':
            continue
        # print('...', area)
        js = CsvFile.load(f)
        if not js.x:

            continue
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
    js = CsvFile.load(csvfile(area='world'))  # load world-level and copy metadata
    data['_metadata'] = js.get_metadata()

    return data

def ranking_file(study_path, variable):
    # return os.path.join(cube_path, indicator, 'ranking.{}.{}.{}.json'.format(indicator, category, variable))
    return os.path.join(study_path, 'ranking.{}.json'.format(variable))


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
        return self.data[area.lower()][scenario or 'null'][index]

    def values(self, x, scenario=None):
        index = self.data['_index'].index(x)
        return [self.data[area.lower()][scenario or 'null'][index] for area in self.areas]

    def sorted_areas(self, x, scenario=None):
        values = self.values(x, scenario)
        pairs = sorted([(v if v is not None else -99e9, area) for v, area in zip(values, self.areas)], key=lambda x: x[0])
        return [area for v, area in pairs[::-1]]

    def number(self, area, x, scenario=None):
        if self.value(area, x, scenario) is None:
            return 'undefined'
        return self.sorted_areas(x, scenario).index(area.lower()) + 1

    def position(self, area, x, scenario=None):
        if self.value(area, x, scenario) is None:
            return 'undefined'
        num = self.number(area, x, scenario)
        return ordinal(num)


    @classmethod
    def load(cls, fname):
        ranking_data = json.load(open(fname))
        return cls(ranking_data)


class RankingCmd:
    """A container class to contain various Ranking instance
    """
    def __init__(self, data=None, area=None):
        self.area = area
        self.data = data or {}

    def __call__(self, variable, x=None, area=None, method='position', **kwargs):
        """select the appropriate ranking class and pass on relevant arguments.
        area defaults to context-specific area
        """
        r = self.data[variable]  # Ranking instance
        func = getattr(r, method) #  Ranking instance method
        kwargs['x'] = x
        if method in ('value', 'number', 'position'):
            kwargs['area'] = area or self.area
        return func(**kwargs)


def preprocess_ranking(cfg, country_names=None):
    for name in cfg.get('ranking-files',[]):
        print('ranking preprocessing:', cfg.get('folder'), name)
        data = calculate_ranking(cfg.get('folder'), name, country_names=country_names or cfg.get('area'))
        print('==>>', len(data), 'countries were covered', ", ".join(data))
        fname = ranking_file(cfg.get('folder'), name)
        dname = os.path.dirname(fname)
        if not os.path.exists(dname):
            os.makedirs(dname)
        json.dump(data, open(fname, 'w'))


def load_ranking(cfg):
    ranking_data = {}
    for name in cfg.get('ranking-files', []):
        fname = ranking_file(cfg.get('folder'), name)
        ranking_data[name] = Ranking.load(fname)
    return ranking_data