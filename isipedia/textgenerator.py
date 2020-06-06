"""Generate Text for ISIPedia Project
"""
import json
import subprocess
import os
import glob
import shutil
import jinja2
import logging
import functools
import netCDF4 as nc
import frontmatter

from isipedia.jsonfile import JsonFile
from isipedia.country import Country, countrymasks_folder, country_data_folder
from isipedia.ranking import load_indicator_config, ranking_file, preprocess_ranking, Ranking
from isipedia.figure import MapData
from isipedia.command import contexts_register, commands_register, figures_register
from isipedia.web import Study, Article, country_codes as allcountries, fix_metadata


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
    def __init__(self, indicator, studytype, area, cube_folder='dist', config=None, ranking=None, makefig=True, variables=None, png=False):
        self.indicator = indicator
        self.studytype = studytype
        self.area = area
        self.cube_folder = cube_folder
        self.config = config or load_indicator_config(indicator)
        self.config['area'] = area
        #self.folder = os.path.join(cube_folder, indicator, studytype, area)
        self.study = Study(**self.config)
        self.folder = os.path.join(cube_folder, self.study.url, area.lower())
        if ranking:
            ranking.area = area # predefine area 
        self.ranking = ranking
        self.makefig = makefig
        self.png = png
        self.variables = variables or {}

    def jsonfile(self, name):
        return os.path.join(self.cube_folder, self.study.url, self.area.lower(), name+'_'+self.area+'.json')

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

    @property
    def metadata(self):
        kw = self.config.copy()
        fix_metadata(kw)
        return kw

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


def load_template_context(indicator, study_type, area, cube_folder='dist', **kwargs):
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

def load_ranking(indicator, cube_folder='dist'):
    cfg = load_indicator_config(indicator)
    ranking = MultiRanking()
    for name in cfg.get('ranking-files', []):
        study_path = os.path.join(cube_folder, Study(**cfg).url)
        fname = ranking_file(study_path, name)
        if not os.path.exists(fname):
            logging.warning('ranking file does not exist: '+fname)
            continue
        ranking[name.replace('-','_')] = Ranking.load(fname)
    return ranking


def process_indicator(indicator, cube_folder, country_names=None, 
    templatesdir='templates', fail_on_error=False, makefig=True, png=False, javascript=None):
  
    cfg = load_indicator_config(indicator)
    study_type = cfg['studytype']

    # Going though all the countries in the list.
    if country_names is None:
        country_names = allcountries

    # load country ranking
    ranking = load_ranking(indicator, cube_folder)

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
        context = load_template_context(indicator, study_type, area, cube_folder, config=cfg, ranking=ranking, makefig=makefig, png=png)

        # add global context
        if makefig:
            context.mapdata = mapdata
            context.countrymasksnc = countrymasksnc
            context.countries = countries
            context.countries_simple = countries_simple

        # extend markdown context with custom values
        for f in contexts_register:
            f(context)

        tmplfile = select_template(indicator, area, templatesdir=templatesdir)
        tmpl = jinja2.Template(open(tmplfile).read())

        os.makedirs(context.folder, exist_ok=True)

        kwargs = context.template_kwargs()

        text = tmpl.render(**kwargs)

        md_file = os.path.join(context.folder, '.{indicator}_{area}.md'.format(indicator=indicator, area=area))

        post = frontmatter.Post(text, **context.metadata)
        frontmatter.dump(post, md_file)
        #with open(md_file, 'w') as f:
        #    f.write(text)

        # copy along javascript?
        javascript2 = javascript or []

        base, ext = os.path.splitext(tmplfile)
        candidatejs = base + '.js'
        if candidatejs not in javascript2 and os.path.exists(candidatejs):
            javascript2 = [candidatejs] + javascript2
        for jsfile in javascript2:
            shutil.copy(jsfile, context.folder)

        return md_file


    md_files = []

    for area in country_names:
        print(indicator+ " - " +area)  
        try:
            md_file = process_area(area)
            md_files.append(md_file)
        except Exception as error:
            if fail_on_error:
                raise
            else:
                logging.warning(str(error))
                print('!! failed',area)

    if makefig:
        countrymasksnc.close()

    return md_files


def main():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--study-types', nargs='*', help='scan all study types by default')
    parser.add_argument('--areas', nargs='*', help='by default: use area field from yaml config')
    parser.add_argument('--indicators', nargs='+', required=True, help='')
    parser.add_argument('--cube-path', default='dist', help='%(default)s')
    parser.add_argument('--ranking', action='store_true', help='preprocess ranking')
    parser.add_argument('--makefig', action='store_true', help='make figures')
    parser.add_argument('--png', action='store_true', help='store interactive figs to png as well (for markdown rendering)')
    parser.add_argument('--no-markdown', action='store_true', help='stop after preprocessing')
    parser.add_argument('--templates-dir', default='templates', help='templates directory (default: %(default)s)')
    parser.add_argument('--skip-error', action='store_true', help='skip area with error instead of raising exception')
    parser.add_argument('--js', nargs='+', default=[], help='additional javascript to be copied along in the folder')
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--pdf', action='store_true', help='make pdf version when building')
    parser.add_argument('--deploy', action='store_true', help='deploy to local isipedia.org')
    parser.add_argument('--deploy-test', action='store_true')
    parser.add_argument('--delete', action='store_true')
    # parser.add_argument('--deploy-demo', action='store_true')
    # parser.add_argument('--deploy-isipedia', action='store_true')

    o = parser.parse_args()

    if os.path.exists('custom.py'):
        print('custom.py module present in work directory. Load it.')
        try:
            import custom
        except ImportError as error:
            raise
            logging.warning('failed to load custom.py module')
            logging.warning(str(error))

    country_data_folder = os.path.join(countrymasks_folder, 'country_data')
    print('country_data:', country_data_folder)

    all_md_files = []
    studies = []

    for indicator in o.indicators:

        if indicator.endswith('.yml'):
            indicator, _ = os.path.splitext(indicator)

        cfg = load_indicator_config(indicator)
        study = Study(**cfg)
        studies.append(study)

        if not o.areas:
            o.areas = study.area

        if o.ranking:
            study_path = os.path.join(o.cube_path, study.url)
            preprocess_ranking(cfg, study_path)
            if o.no_markdown:
                pass

        try:
            md_files = process_indicator(indicator, o.cube_path+'/', country_names=o.areas, 
                templatesdir=o.templates_dir, fail_on_error=not o.skip_error, makefig=o.makefig, png=False, javascript=o.js)
            all_md_files.extend(md_files)
        except Exception as error:
            raise
            print(error)
            continue

    if o.build:
        from isipedia.web import root
        import sys
        cmd = [sys.executable,os.path.join(root, 'scripts', 'process_articles.py'), '--update','--out', o.cube_path, '--html'] + md_files
        if o.png: cmd += ['--png']
        if o.pdf: cmd += ['--pdf']
        print(' '.join(cmd))
        subprocess.run(cmd)


    def deploy(root):
        for study in studies:
            cmd = ['rsync','-avzr', os.path.join(o.cube_path, study.url)+'/', f'{root}/{study.url}/']
            if o.delete:
                cmd.apend('--delete')
            print(' '.join(cmd))
            subprocess.run(cmd)

        # cmd = ['rsync','-avzr', o.cube_path+'/pdf/', f'{root}/pdf/']
        # # cmd = ['rsync','-avzr', os.path.join(o.cube_path, study.url_pdf), os.path.join(root, 'dist', study.url_pdf)]
        # # cmd = ['cp', os.path.join(o.cube_path, study.url).replace('report', 'pdf')+'*', os.path.join(root, 'dist', 'pdf')]
        # print(' '.join(cmd))
        # subprocess.run(cmd)


    if o.deploy:
        from isipedia.web import root
        deploy(root / 'dist')


    if o.deploy_test:
        import socket
        host = socket.gethostname()
        if host.startswith('login'):
            server = 'se59:'
        else:
            server = 'isipedia.org:'
        remote = server+'/webservice/test.isipedia.org'
        deploy(remote)


if __name__ == '__main__':
    main()    
