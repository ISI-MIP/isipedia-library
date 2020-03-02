"""Generate Text for ISIPedia Project
"""
import json
import os
import glob
import jinja2
import logging
import netCDF4 as nc

from isipedia.jsonfile import JsonFile
from isipedia.figure import LinePlot, CountryMap, RankingMap, MapData
from isipedia.country import CountryStats, countrymasks_folder


class CubeVariable:
    """This class corresponds to cube definition of a variable
    """
    def __init__(self, indicator, studytype, name):
        self.indicator = indicator
        self.studytype = studytype
        self.name = name

    # def getmodels(self, cube_path, area=None):
    #     """load models list for that variable (should be area independent)
    #     """
    #     fname = glob.glob(self.jsonfile(area or '*', cube_path))[0]
    #     js = json.load(open(fname))
    #     return {'climate_model_list': js['climate_model_list'], 'impact_model_list': js['impact_model_list']}

    def jsonfile(self, area, cube_path):
        return os.path.join(cube_path, self.indicator, self.studytype, area, self.name+'_'+area+'.json')

    @staticmethod
    def _varname(fname, area=None):
        ' determine variable name from json file name '
        name, ext = os.path.splitext(os.path.basename(fname))
        if name.endswith(area):
            name = name[:-len(area)-1]
        return name.replace('-','_')


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


def load_indicator_config(indicator, cube_path):
    cfgfile = os.path.join(cube_path, indicator, 'config.json')
    if not os.path.exists(cfgfile):
        logging.warn('no config file present for '+indicator)
    return json.load(open(cfgfile))


def ranking_file(indicator, category, variable, cube_path):
    # return os.path.join(cube_path, indicator, 'ranking.{}.{}.{}.json'.format(indicator, category, variable))
    return os.path.join(cube_path, indicator, category, 'world', 'ranking', 'ranking.{}.json'.format(variable))


def preprocess_ranking(indicator, cube_path, out_cube_path=None, country_names=None):
    if out_cube_path is None:
        out_cube_path = cube_path

    cfg = load_indicator_config(indicator, cube_path)

    for study in cfg['study-types']:
        for name in study['ranking-files']:
            category = study['directory']
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


class TemplateData(CountryStats):
    """template data accessible for an author
    """
    def __init__(self, code, variables, **stats):
        super().__init__(**stats)
        self.code = code
        self.variables = variables

    def __getattr__(self, name):
        return self.variables[name]

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



def load_country_data(indicator, study_type, area, input_folder, country_data_folder=None):
    jsfiles = glob.glob(CubeVariable(indicator, study_type, '*').jsonfile(area, input_folder))
    variables = {CubeVariable._varname(fname, area):JsonFile.load(fname) for fname in jsfiles}
    if country_data_folder is None:
        country_data_folder = 'cube/country_data'
    try:
        stats = CountryStats.load(os.path.join(country_data_folder, area, area+'_general.json'))
        # stats = CountryStats(area)
    except Exception as error:
        print('!!', str(error))
        raise
        logging.warning("country stats not found for: "+area)
        # raise
        stats = CountryStats("undefined")

    return TemplateData(area, variables=variables, **vars(stats))


def select_template(indicator, area=None, templatesdir='templates'):
    candidates = [
        '{templatesdir}/{indicator}/{indicator}_{area}.md'.format(indicator=indicator, area=area, templatesdir=templatesdir),
        '{templatesdir}/{indicator}/{indicator}.md'.format(indicator=indicator, templatesdir=templatesdir),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    print('\n'.join(candidates))
    raise ValueError('no template found for '+repr((indicator, area)))


# from jinja2 import Environment, FileSystemLoader, select_autoescape
# env = Environment(
#     loader=FileSystemLoader('./'),
#     autoescape=select_autoescape(['html', 'xml'])
# )


    # def map(self, x, scenario=None, **kwargs):
    #     return RankingMap(self, x, scenario, **kwargs)


class MultiRanking(dict):
    def __init__(self, ranking=None, area=None):
        self.area = area 
        super().__init__(ranking or {})

    def __call__(self, variable, x=None, area=None, method='position', **kwargs):
        """select the appropriate ranking class and pass on relevant arguments.
        area defaults to context-specific area
        """
        r = self[variable]  # Ranking instance
        func = getattr(r, method) #  Ranking instance method
        kwargs['x'] = x
        if method in ('value', 'number', 'position'):
            kwargs['area'] = area or self.area
        return func(**kwargs)


    # def map(self, variable, x=None, scenario=None, title='', **kwargs):
    #     return self[variable].map(x, scenario, title=title or variable, **kwargs)



class StudyType:
    def __init__(self, code, name='', description=''):
        self.code = code
        self.name = name or code.replace('-',' ').capitalize()
        self.description = description or ''


class Indicator:
    def __init__(self, code, name):
        self.code = code
        self.name = name


def process_indicator(indicator, input_folder, output_folder, country_names=None, study_type='future-projections',
    templatesdir='templates', country_data_folder=None, fail_on_error=False, backend='mpl', makefig=True, backends=None):
  
    world = load_country_data(indicator, study_type, 'world', input_folder, country_data_folder=country_data_folder)

    # Going though all the countries in the list.
    if country_names is None:
        country_names = sorted(os.listdir (os.path.join(input_folder, indicator, study_type)))
        country_names = [c for c in country_names if os.path.exists(os.path.join(country_data_folder, c))]

    cfg = load_indicator_config(indicator, input_folder)

    # Select study
    found = False
    for stype in cfg['study-types']:
        if stype['directory'] == study_type:
            found = True
            break
    if not found:
        raise ValueError('studytype not defined in config.json file: '+repr(study_type))

    # metadata
    meta_indicator = Indicator(indicator, cfg.get('name'))
    meta_studytype = StudyType(study_type, stype.get('name'))

    # load country ranking
    ranking = MultiRanking()
    for name in stype['ranking-files']:
        fname = ranking_file(indicator, study_type, name, output_folder)
        if not os.path.exists(fname):
            logging.warning('ranking file does not exist: '+fname)
            continue
        ranking[name.replace('-','_')] = Ranking.load(fname)

    # used by the figures
    countrymasksnc = nc.Dataset(os.path.join(countrymasks_folder, 'countrymasks.nc'))
    mapdata = MapData(indicator, study_type, input_folder, country_data_folder)

    def process_area(area):
        country = load_country_data(indicator, study_type, area, input_folder, country_data_folder=country_data_folder)
        tmplfile = select_template(indicator, area, templatesdir=templatesdir)
        tmpl = jinja2.Template(open(tmplfile).read())
        # tmpl = env.get_template(tmplfile)
        ranking.area = area # predefine area 

        output_folder_local = os.path.join(output_folder, indicator, study_type, area)
        if not os.path.exists(output_folder_local):
            os.makedirs(output_folder_local)
    
        text = tmpl.render(country=country, world=world, ranking=ranking, 
            lineplot=LinePlot(output_folder_local, backend, makefig, backends=backends), 
            rankingmap=RankingMap(ranking, countrymasksnc, output_folder_local, backend, makefig, backends=backends), 
            countrymap=CountryMap(mapdata, countrymasksnc, output_folder_local, backend, makefig, backends=backends), 
            indicator=meta_indicator, studytype=meta_studytype, **country.variables)

        md_file = os.path.join(output_folder_local, '{indicator}-{area}.md'.format(indicator=indicator, area=area))
        with open(md_file, 'w') as f:
            f.write(text)

    for area in country_names:
        print(indicator+ " - " +area)  
        try:
            process_area(area)
        except Exception as error:
            if fail_on_error:
                raise
            else:
                logging.warning(str(error))
                print('!! failed',area)

    countrymasksnc.close()


def main():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--study-types', nargs='*', help='scan all study types by default')
    parser.add_argument('--areas', nargs='*', help='scan all areas by default')
    parser.add_argument('--indicators', nargs='+', required=True, help='')
    parser.add_argument('--cube-path', default='cube', help='%(default)s')
    parser.add_argument('--out-cube-path', help='if output shall differs from output')
    parser.add_argument('--ranking', action='store_true', help='preprocess ranking')
    parser.add_argument('--makefig', action='store_true', help='make figures')
    parser.add_argument('--backend', default='mpl', help='default backend for figures')
    parser.add_argument('--backends', default=None, help='backends to crunch')
    parser.add_argument('--no-markdown', action='store_true', help='stop after preprocessing')
    parser.add_argument('--templates-dir', default='templates', help='templates directory (default: %(default)s)')
    parser.add_argument('--skip-error', action='store_true', help='skip area with error instead of raising exception')

    o = parser.parse_args()


    country_data_folder = os.path.join(countrymasks_folder, 'country_data')
    print('country_data:', country_data_folder)

    if not o.out_cube_path:
        o.out_cube_path = o.cube_path

    for indicator in o.indicators:

        if o.ranking:
            preprocess_ranking(indicator, o.cube_path, o.out_cube_path)
            if o.no_markdown:
                pass

        if not o.study_types:
            study_types = [d for d in os.listdir(os.path.join(o.cube_path, indicator)) if os.path.isdir(os.path.join(o.cube_path, indicator, d))]
        else:
            study_types = o.study_types

        for studytype in study_types:
            print(studytype)
            try:
                process_indicator(indicator, o.cube_path+'/', o.out_cube_path+'/', country_names=o.areas, 
                    study_type=studytype, templatesdir=o.templates_dir, country_data_folder=country_data_folder,
                    fail_on_error=not o.skip_error, makefig=o.makefig, backend=o.backend, backends=o.backends)
            except Exception as error:
                raise
                print(error)
                continue


if __name__ == '__main__':
    main()    
