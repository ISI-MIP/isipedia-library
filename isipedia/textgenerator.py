"""Generate Text for ISIPedia Project
"""
import json
import os
import glob
import shutil
import jinja2
import logging
import functools
import netCDF4 as nc

from isipedia.jsonfile import JsonFile
from isipedia.country import Country, countrymasks_folder, country_data_folder
from isipedia.ranking import load_indicator_config, ranking_file, preprocess_ranking, Ranking
from isipedia.figure import MapData
from isipedia.command import contexts_register, commands_register, figures_register
from isipedia.web import Study, Article


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


class StudyType:
    def __init__(self, code, name='', description=''):
        self.code = code
        self.name = name or code.replace('-',' ').capitalize()
        self.description = description or ''


class Indicator:
    def __init__(self, code, name):
        self.code = code
        self.name = name


class TemplateContext:
    """template data accessible within jinja2 and provided to various functions such as figures
    """
    def __init__(self, indicator, studytype, area, cube_folder='cube', config=None, ranking=None):
        self.indicator = indicator
        self.studytype = studytype
        self.area = area
        self.cube_folder = cube_folder
        self.config = config or load_indicator_config(indicator)
        #self.folder = os.path.join(cube_folder, indicator, studytype, area)
        self.study = Study(**self.config)
        self.folder = os.path.join(cube_folder, self.study.url, area)
        if ranking:
            ranking.area = area # predefine area 
        self.ranking = ranking


    def jsonfile(self, name):
        return os.path.join(self.cube_folder, self.indicator, self.studytype, self.area, name+'_'+self.area+'.json')

    def _simplifyname(self, fname):
        ' determine variable name from json file name '
        name, ext = os.path.splitext(os.path.basename(fname))
        if name.endswith(self.area):
            name = name[:-len(self.area)-1]
        return name.replace('-','_')

    def load_json_files(self):
        jsfiles = glob.glob(self.jsonfile('*'))
        self.variables = {self._simplifyname(fname):JsonFile.load(fname) for fname in jsfiles}

    def load_country_stats(self):
        try:
            self.country = Country.load(os.path.join(country_data_folder, self.area, self.area+'_general.json'))
            # stats = CountryStats(self.area)
        except Exception as error:
            print('!!', str(error))
            logging.warning("country stats not found for: "+self.area)
            # raise
            self.country = Country("undefined")


    def template_kwargs(self):
        figure_functions = {name:cls(self) for name, cls in figures_register.items()}
        markdown_commands = {name:functools.partial(func, self) for name, func in commands_register.items()}
        kwargs = self.variables.copy()
        kwargs.update(figure_functions)
        kwargs.update(markdown_commands)
        kwargs.update(dict(
            country=self.country, 
            indicator=self.indicator,
            studytype=self.studytype,
            config=self.config,
            ranking=self.ranking,
            ))
        return kwargs


    def __getattr__(self, name):
        return self.variables[name]


def load_template_context(indicator, study_type, area, cube_folder='cube', **kwargs):
    context = TemplateContext(indicator, study_type, area, cube_folder, **kwargs)
    context.load_json_files()
    context.load_country_stats()
    return context


def select_template(indicator, area=None, templatesdir='templates'):
    candidates = [
        '{templatesdir}/{indicator}/{indicator}_{area}.md'.format(indicator=indicator, area=area, templatesdir=templatesdir),
        '{templatesdir}/{indicator}/{indicator}.md'.format(indicator=indicator, templatesdir=templatesdir),
        '{templatesdir}/{indicator}_{area}.md'.format(indicator=indicator, area=area, templatesdir=templatesdir),
        '{templatesdir}/{indicator}.md'.format(indicator=indicator, templatesdir=templatesdir),
        '{indicator}_{area}.md'.format(indicator=indicator, area=area, templatesdir=templatesdir),
        '{indicator}.md'.format(indicator=indicator, templatesdir=templatesdir),
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

    # def map(self, variable, x=None, scenario=None, title='', **kwargs):
    #     return self[variable].map(x, scenario, title=title or variable, **kwargs)


def process_indicator(indicator, cube_folder, country_names=None, study_type='future-projections',
    templatesdir='templates', fail_on_error=False, makefig=True, png=False, javascript=None):
  
    # Going though all the countries in the list.
    if country_names is None:
        country_names = sorted(os.listdir (os.path.join(cube_folder, indicator, study_type)))
        country_names = [c for c in country_names if os.path.exists(os.path.join(country_data_folder, c))]

    cfg = load_indicator_config(indicator)

    # load country ranking
    ranking = MultiRanking()
    for name in cfg.get('ranking-files', []):
        fname = ranking_file(indicator, study_type, name, cube_folder)
        if not os.path.exists(fname):
            logging.warning('ranking file does not exist: '+fname)
            continue
        ranking[name.replace('-','_')] = Ranking.load(fname)

    # used by the figures
    if makefig:
        countrymasksnc = nc.Dataset(os.path.join(countrymasks_folder, 'countrymasks.nc'))
        countries = json.load(open(countrymasks_folder+'/countrymasks.geojson'))['features']
        countries_simple = json.load(open(countrymasks_folder+'/countrymasks.geojson'))['features']
        import shapely.geometry as shg
        import shapely.ops
        logging.info('simplify countries geometry for ranking map')
        for c in countries_simple:
            # simplify for faster rendering
            simple = shg.shape(c['geometry']).simplify(0.1) 
            simple2 = shapely.ops.transform(lambda x, y: (round(x, 2), round(y, 2)), simple)
            c['geometry'] = shg.mapping(simple2)

        mapdata = MapData(indicator, study_type, cube_folder)


    def process_area(area):
        context = load_template_context(indicator, study_type, area, cube_folder, config=cfg, ranking=ranking)

        # add global context
        if makefig:
            context.countrymasksnc = countrymasksnc
            context.countries = countries
            context.countries_simple = countries_simple
            context.mapdata = mapdata
            context.makefig = makefig
            context.png = png

        # extend markdown context with custom values
        for f in contexts_register:
            f(context)

        tmplfile = select_template(indicator, area, templatesdir=templatesdir)
        tmpl = jinja2.Template(open(tmplfile).read())

        os.makedirs(context.folder, exist_ok=True)

        kwargs = context.template_kwargs()

        text = tmpl.render(**kwargs)

        md_file = os.path.join(context.folder, '{indicator}-{area}.md'.format(indicator=indicator, area=area))
        with open(md_file, 'w') as f:
            f.write(text)

        # copy along javascript?
        javascript2 = javascript or []

        base, ext = os.path.splitext(tmplfile)
        candidatejs = base + '.js'
        if candidatejs not in javascript2 and os.path.exists(candidatejs):
            javascript2 = [candidatejs] + javascript2
        for jsfile in javascript2:
            shutil.copy(jsfile, context.folder)


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

    if makefig:
        countrymasksnc.close()


def main():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--study-types', nargs='*', help='scan all study types by default')
    parser.add_argument('--areas', nargs='*', help='scan all areas by default')
    parser.add_argument('--indicators', nargs='+', required=True, help='')
    parser.add_argument('--cube-path', default='dist', help='%(default)s')
    parser.add_argument('--ranking', action='store_true', help='preprocess ranking')
    parser.add_argument('--makefig', action='store_true', help='make figures')
    parser.add_argument('--png', action='store_true', help='store interactive figs to png as well (for markdown rendering)')
    parser.add_argument('--no-markdown', action='store_true', help='stop after preprocessing')
    parser.add_argument('--templates-dir', default='templates', help='templates directory (default: %(default)s)')
    parser.add_argument('--skip-error', action='store_true', help='skip area with error instead of raising exception')
    parser.add_argument('--js', nargs='+', default=[], help='additional javascript to be copied along in the folder')

    o = parser.parse_args()

    if os.path.exists('custom.py'):
        print('custom.py module present in work directory. Load it.')
        try:
            import custom
        except ImportError as error:
            logging.warning('failed to load custom.py module')
            logging.warning(str(error))

    country_data_folder = os.path.join(countrymasks_folder, 'country_data')
    print('country_data:', country_data_folder)

    for indicator in o.indicators:

        if o.ranking:
            preprocess_ranking(indicator, o.cube_path)
            if o.no_markdown:
                pass

        if not o.study_types:
            study_types = [d for d in os.listdir(os.path.join(o.cube_path, indicator)) if os.path.isdir(os.path.join(o.cube_path, indicator, d))]
        else:
            study_types = o.study_types

        for studytype in study_types:
            print(studytype)
            try:
                process_indicator(indicator, o.cube_path+'/', country_names=o.areas, 
                    study_type=studytype, templatesdir=o.templates_dir, fail_on_error=not o.skip_error, makefig=o.makefig, png=o.png, javascript=o.js)
            except Exception as error:
                raise
                print(error)
                continue


if __name__ == '__main__':
    main()    
