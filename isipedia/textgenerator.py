"""Generate Text for ISIPedia Project
"""
import json
import os
import glob
import jinja2
import logging


class JsonData:
    """ This class is a useful abstraction of a JsonFile that 
    can accomodate for both vs-time and vs-temp data structure with a minimum of code modifications
    """
    def __init__(self, data, x, vs_time=False):
        self.data = data
        self.x = x
        self.vs_time = vs_time

    def _data(self, scenario):
        return self.data[scenario] if self.vs_time else self.data

    def index(self, x):
        time_map = {
          "farfuture" : "2081-2100",
          "nearfuture" : "2041-2060",
          "today" : "2001-2020",
        }
        return self.x.index(time_map.get(x, x)) 

    def getarray(self, scenario=None, climate_model=None, impact_model=None, field=None):
        """get array of x*scenario """
        data = self._data(scenario)
        if impact_model:
          return data[climate_model or 'overall']['runs'][impact_model][field or 'mean']
        else:
          return data[climate_model or 'overall'][field or 'median']

    def get(self, x, scenario=None, climate_model=None, impact_model=None, field=None):
        """get single number """
        index = self.index(x)
        y = self.getarray(scenario, climate_model, impact_model, field)
        return y[index]


    def getens(self, x, scenario=None, climate_model=None, impact_model=None, field='mean'):
        """get an array of values for all climate and impact models
        """
        index = self.index(x)
        data = self._data(scenario)
        climate_models = self.climate_model_list if climate_model is None else [climate_model]
        impact_models = self.impact_model_list if impact_model is None else [impact_model]
        alldata = [data[climate_model]['runs'][impact_model][field][index] for climate_model in climate_models for impact_model in impact_models]
        return [l for l in alldata if l is not None]

    getall = getens  # alias for backcompatibility


class JsonFile(JsonData):
    """This subclass is in fact a wrapper around JsonData
    """
    def __init__(self, fname):
        self._filename = fname
        js = json.load(open(fname))
        vars(self).update(js)

        # structure similar to JsonData
        if self.plot_type == 'indicator_vs_temperature':
          super().__init__(js['data'], self.temperature_list)
        elif self.plot_type == 'indicator_vs_timeslices':
          x = ['{}-{}'.format(y1, y2) for y1, y2 in self.timeslices_list]
          super().__init__(js['data'], x, vs_time=True)
        elif self.plot_type == 'indicator_vs_timeline':
          # super().__init__(js['data'], self.year_list, vs_time=True)
          logging.info('skip timeline files')
          return
        else:
          logging.warning('unknown file type: '+repr(fname))
          return

        # # one more level of nesting, define scenarios as attributes
        # else:
        #   self._data = {scenario:JsonData(self.data[scenario], x) for scenario in self.climate_scenario_list}
        #   for scenario in self._data:
        #     setattr(self, scenario, self._data[scenario])


class CubeVariable:
    """This class corresponds to cube definition of a variable
    """
    def __init__(self, indicator, studytype, name):
        self.indicator = indicator
        self.studytype = studytype
        self.name = name

    def getmodels(self, cube_path, area=None):
        """load models list for that variable (should be area independent)
        """
        fname = glob.glob(self.jsonfile(area or '*', cube_path))[0]
        js = json.load(open(fname))
        return {'climate_model_list': js['climate_model_list'], 'impact_model_list': js['impact_model_list']}

    def jsonfile(self, area, cube_path):
        return os.path.join(cube_path, self.indicator, self.studytype, area, self.name+'_'+area+'.json')

    @staticmethod
    def _varname(fname, area=None):
        ' determine variable name from json file name '
        name, ext = os.path.splitext(os.path.basename(fname))
        return name.replace('-','_').rstrip('_{}'.format(area)) # remove area name


class CountryStats:
    """This is the class for the corresponding json file in country_data 
    """
    def __init__(self, name, type="country", sub_countries=[], stats=None):
        self.name = name
        self.type = type
        self.sub_countries = sub_countries
        self.stats = stats or []

    @property
    def pop_total(self):
        "million people"
        i = [e['type'] for e in self.stats].index('POP_TOTL')
        return self.stats[i]['value']

    @property
    def pop_density(self):
        'people/km2'
        i = [e['type'] for e in self.stats].index('POP_DNST')
        return self.stats[i]['value']

    @property
    def area(self):
        "km2"
        return self.pop_total*1e6 / self.pop_density

    # @property
    # def pop_rural_pct(self):
    #   i = [e['type'] for e in self.stats].index('RUR_POP_PRCT')
    #   return self.stats[i]['value']
    
    # @property
    # def pop_urban_pct(self):
    #   i = [e['type'] for e in self.stats].index('URB_POP_PRCT')
    #   return self.stats[i]['value']

    @classmethod
    def load(cls, fname):
        js = json.load(open(fname))
        return cls(js['name'], js['type'], js['sub-countries'], stats=js['stats'])


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

    def load_country_data(indicator, study_type, area, input_folder, country_data_folder='cube/country_data'):
        jsfiles = glob.glob(CubeVariable(indicator, study_type, '*').jsonfile(area, input_folder))
        variables = {CubeVariable._varname(fname, area):JsonFile(fname) for fname in jsfiles}
        try:
            stats = CountryStats.load(os.path.join(country_data_folder, area, area+'_general.json'))
        except:
            logging.warning("country stats not found for: "+area)
            raise
            stats = CountryStats("undefined")


        # for some reason all numbers are string, convert to float
        for e in stats.stats:
            try:
                e['value'] = float(e['value'])
            except:
                raise
                pass
            try:
                e['rank'] = float(e['rank'])
            except:
                raise
                pass

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

def _load_ranking_data(var, cube_path, country_names=None):
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
        js = JsonFile(f)
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
    data['_plot_type'] = js.plot_type

    return data


def load_ranking_config(indicator, cube_path):
    cfgfile = os.path.join(cube_path, indicator, 'config.json')
    if not os.path.exists(cfgfile):
        logging.warn('no config file present for '+indicator)
    return json.load(open(cfgfile))


def ranking_file(indicator, category, variable, cube_path):
    return os.path.join(cube_path, indicator, 'ranking.{}.{}.{}.json'.format(indicator, category, variable))


def preprocess_ranking(indicator, cube_path, out_cube_path=None, country_names=None):
    if out_cube_path is None:
        out_cube_path = cube_path

    cfg = load_ranking_config(indicator, cube_path)

    for study in cfg['study-types']:
        for name in study['ranking-files']:
            category = study['directory']
            print(indicator, category, name)
            var = CubeVariable(indicator, category, name)
            data = _load_ranking_data(var, cube_path, country_names=country_names)
            fname = ranking_file(indicator, category, name, out_cube_path)
            dname = os.path.dirname(fname)
            if not os.path.exists(dname):
                os.makedirs(dname)
            json.dump(data, open(fname, 'w'))


class Ranking:
    def __init__(self, data):
        self.data = data
        self.areas = sorted([area for area in data if not area.startswith('_')])

    def value(self, area, x, scenario=None):
        index = self.data['_index'].index(x)
        if self.data['_plot_type'] == 'indicator_vs_temperature':
            return self.data[area]['null'][index]
        else:
            return self.data[area][scenario][index]

    def values(self, x, scenario=None):
        index = self.data['_index'].index(x)
        if self.data['_plot_type'] == 'indicator_vs_temperature':
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


class MultiRanking(dict):
    def __getattr__(self, att):
        return self[att]


def ordinal(num):
    if 10 <= num % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1:'st', 2:'nd', 3:'rd'}.get(num % 10, 'th')
    return '{}{}'.format(num, suffix)


def process_indicator(indicator, input_folder, output_folder, country_names=None, study_type='ISIMIP-projections', 
    templatesdir='templates'):
  
    world = load_country_data(indicator, study_type, 'world', input_folder)

    # Going though all the countries in the list.
    if country_names is None:
        country_names = sorted(os.listdir (os.path.join(input_folder, indicator, study_type)))

    # load country ranking
    cfg = load_ranking_config(indicator, output_folder)
    ranking = MultiRanking()
    for study in cfg['study-types']:
        if study['directory'] == study_type:
            for name in study['ranking-files']:
                fname = ranking_file(indicator, study_type, name, output_folder)
                if not os.path.exists(fname):
                    logging.warning('ranking file does not exist: '+fname)
                    continue
                ranking_data = json.load(open(fname))
                ranking[name.replace('-','_')] = Ranking(ranking_data)


    for area in country_names:
        print(indicator+ " - " +area)  

        country = load_country_data(indicator, study_type, area, input_folder)
        tmplfile = select_template(indicator, area, templatesdir=templatesdir)
        tmpl = jinja2.Template(open(tmplfile).read())
        # tmpl = env.get_template(tmplfile)
        text = tmpl.render(country=country, world=world, ranking=ranking, **country.variables)

        output_folder_local = os.path.join(output_folder, indicator, study_type, area)
        if not os.path.exists(output_folder_local):
            os.makedirs(output_folder_local)
    
        md_file = os.path.join(output_folder_local, '{indicator}-{area}.md'.format(indicator=indicator, area=area))
        with open(md_file, 'w') as f:
            f.write(text)


def main():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('indicator_name')
    parser.add_argument('--study-types', nargs='*', help='scan all study types by default')
    parser.add_argument('--areas', nargs='*', help='scan all areas by default')
    parser.add_argument('--cube-path', default='cube', help='%(default)s')
    parser.add_argument('--out-cube-path', help='if output shall differs from output')
    parser.add_argument('--ranking', action='store_true', help='preprocess ranking')
    parser.add_argument('--no-markdown', action='store_true', help='stop after preprocessing')
    parser.add_argument('--templates-dir', default='templates', help='templates directory (default: %(default)s)')

    o = parser.parse_args()

    if not o.out_cube_path:
        o.out_cube_path = o.cube_path

    if o.ranking:
        preprocess_ranking(o.indicator_name, o.cube_path, o.out_cube_path)
        if o.no_markdown:
            pass

    if not o.study_types:
        study_types = [d for d in os.listdir(os.path.join(o.cube_path, o.indicator_name)) if os.path.isdir(os.path.join(o.cube_path, o.indicator_name, d))]
    else:
        study_types = o.study_types

    for studytype in study_types:
        print(studytype)
        try:
            process_indicator(o.indicator_name, o.cube_path+'/', o.out_cube_path+'/', country_names=o.areas, 
                study_type=studytype, templatesdir=o.templates_dir)
        except Exception as error:
            raise
            print(error)
            continue


if __name__ == '__main__':
    main()    
