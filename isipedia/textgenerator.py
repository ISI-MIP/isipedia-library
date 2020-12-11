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
import frontmatter
import yaml
from normality import slugify


from isipedia.jsonfile import JsonFile, CsvFile
from isipedia.country import Country, country_data_folder
from isipedia.ranking import preprocess_ranking, load_ranking, RankingCmd
from isipedia.command import study_context_register, contexts_register, commands_register, figures_register
from isipedia.web import country_codes as allcountries, country_names, fix_metadata

allcountries = sorted(allcountries)

class NameSpace:
    def __init__(self, **kwargs):
        vars(self).update(kwargs)

    def get(self, name, alt=None):
        return getattr(self, name, alt)


class StudyConfig(NameSpace):
    def __init__(self, title=None, author=None, area=None, institution=None,
            topics=None, studytype=None, published=None, doi=None, beta=False, indicator=None, root='dist',
            skip=False, short_name=None, story=False, **kwargs):

        self.title = title
        self.author = author or []
        self.area = area or (allcountries if not story else '')
        self.institution = institution or []
        self.studytype = studytype
        self.topics = topics
        self.published = published
        self.story = story
        self.doi = doi
        self.beta = beta
        self.root = root
        self.skip = skip
        self.indicator = indicator or os.path.basename(self.url)
        self.short_name = short_name or os.path.basename(os.path.abspath(''))  # directory under isipedia-studies
        vars(self).update(kwargs) # also accepts fields like 'ranking-files'

    def __iter__(self):
        ' so that "field in config" works'
        return iter(vars(self))

    @property
    def stem(self):
        return 'story' if self.story else 'report'

    @property
    def url(self):
        study_url = self.stem + '/'+slugify(self.title)
        if self.area is None or type(self.area) is list:
            return study_url
        else:
            return study_url +'/'+self.area.lower()

    @property
    def folder(self):
        return os.path.join(self.root, self.url)+ '/'

    @classmethod
    def load(cls, fname, **kwargs):
        indicator, _ = os.path.splitext(fname)
        cfgfile = os.path.join(indicator+'.yml')
        # if not os.path.exists(cfgfile):
        #     raise ValueError('no config file present for '+indicator)
        cfg = yaml.safe_load(open(cfgfile))
        cfg['indicator'] = indicator
        cfg.update(kwargs)
        if not cfg.get('story'):
            areas = [area for area in cfg.get('area', allcountries) if area not in cfg.get('exclude-countries',[])]
            cfg['area'] = areas
        return cls(**cfg)


def load_country_stats(area):
    try:
        country = Country.load(os.path.join(country_data_folder, area, area+'_general.json'))
        # stats = CountryStats(area)
    except Exception as error:
        print('!!', str(error))
        logging.warning("country stats not found for: "+area)
        # raise
        country = Country("undefined")
    return country


class TemplateContext(StudyConfig):
    """pass on to jinja2 templates
    """
    def __init__(self, study=None, variables=None, area=None, **data):
        self.study = study or {}
        self.variables = variables or {}
        self.area = area
        vars(self).update(data)


    def __getattr__(self, name):
        # To access study parameter such as title, etc... unless set here (e.g. area)
        if hasattr(self.study, name):
            return getattr(self.study, name)
        elif name in self.variables:
            return self.variables[name]
        else:
            raise AttributeError(name)

    def get(self, name, alt=None):  # call __getattr__ without raising an error
        return getattr(self, name, alt)

    @property
    def markdown(self):
        if self.area:
            return os.path.join(self.folder, f'.source_{self.area}.md')
            # return os.path.join(self.folder, f'.{self.indicator}_{self.area}.md')
        else:
            return os.path.join(self.folder, f'.source.md')

    # legacy
    def jsonfile(self, name, ext=".json"):
        return os.path.join(self.folder, name+'_'+self.area+ext)

    def csvfile(self, name):
        return self.jsonfile(name, ext=".csv")

    def _simplifyname(self, fname):
        ' determine variable name from json file name '
        name, ext = os.path.splitext(os.path.basename(fname))
        if self.area and name.endswith(self.area):
            name = name[:-len(self.area)-1]
        return name.replace('-','_')

    def load_json_files(self):
        jsfiles = glob.glob(self.jsonfile('*'))
        variables = {self._simplifyname(fname):JsonFile.load(fname) for fname in jsfiles}
        if not set(variables).isdisjoint(set(self.variables)):
            logging.warning("variables already loaded")
        self.variables.update(variables)

    def load_csv_files(self):
        jsfiles = glob.glob(self.csvfile('*'))
        # print('csv file pattern', self.csvfile('*'))
        # print('csvfiles', jsfiles)
        variables = {self._simplifyname(fname):CsvFile.load(fname) for fname in jsfiles}
        if not set(variables).isdisjoint(set(self.variables)):
            logging.warning("variables already loaded")
        self.variables.update(variables)

    def load_files(self):
        self.load_json_files()
        self.load_csv_files()
        # self.load_country_stats()

    @property
    def metadata(self):
        kw = vars(self.study).copy()
        kw['area'] = self.area
        if self.area == 'world':
            kw['sub-countries'] = [{'code':a, 'name':country_names[a]} for a in self.study.area]
        fix_metadata(kw)
        return kw

    def template_kwargs(self):
        figure_functions = {name:cls(self) for name, cls in figures_register.items()}
        markdown_commands = {name:functools.partial(func, self) for name, func in commands_register.items()}
        if self.area:
            country = getattr(self, 'country') or load_country_stats(self.area)
        else:
            country = None
        kwargs = self.variables.copy()
        kwargs.update(figure_functions)
        kwargs.update(markdown_commands)
        if self.area:
            kwargs.update({'ranking': RankingCmd(self.get('ranking_data', {}), self.area)})
        kwargs.update(dict(
            country=country,
            indicator=self.indicator,
            studytype=self.studytype,
            config=self.study,
            markdown=self.markdown,
            ))
        return kwargs


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

def process_markdown(context):

    # extend markdown context with custom values
    for f in contexts_register:
        f(context)

    if context.get('setup_only'):
        return context.markdown

    tmplfile = select_template(context.indicator, context.area, templatesdir=context.templates_dir)
    tmpl = jinja2.Template(open(tmplfile).read())

    os.makedirs(context.folder, exist_ok=True)

    kwargs = context.template_kwargs()

    text = tmpl.render(**kwargs)

    md_file = context.markdown

    post = frontmatter.Post(text, **context.metadata)
    frontmatter.dump(post, md_file)



    return md_file


def process_study(study, country_names=None, fail_on_error=False, update=True, **kwargs):

    # execute setup scripts, if any
    for cmd in study.get('setup', []):
        sp.check_call(cmd, shell=True)

    study_context = TemplateContext(study, **kwargs)

    # execute setup commands from custom.py
    for f in study_context_register:
        f(study_context)

    # setup ranking, if required
    if study.get('ranking-files'):
        # load country ranking
        try:
            study_context.ranking_data = load_ranking(study)

        except FileNotFoundError:
            print('preprocess ranking !')
            preprocess_ranking(study)
            study_context.ranking_data = load_ranking(study)

    # Going though all the countries in the list.
    if country_names is None:
        country_names = study.area or [''] # list of study areas (before the loop below we have context.area == context.study.area)

    md_files = []

    default_context_fields = vars(TemplateContext())

    for area in country_names:

        context = TemplateContext(study,
            area=area,
            country=load_country_stats(area) if area else None,
            **{k:v for k,v in vars(study_context).items() if k not in default_context_fields})

        if os.path.exists(context.markdown) and not update:
            md_files.append(context.markdown)
            continue

        print(context.url)

        if context.get('autoload'):
            context.load_files()

        try:
            md_file = process_markdown(context)
            md_files.append(md_file)
        except Exception as error:
            if fail_on_error:
                raise
            else:
                logging.warning(str(error))
                print('!! failed',area)

    return md_files


def main():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--makefig', action='store_true', default=None, help=argparse.SUPPRESS)

    parser.add_argument('--ranking', action='store_true', default=True, help=argparse.SUPPRESS)
    parser.add_argument('--no-ranking', action='store_false', dest='ranking', default=True, help='do not preprocess ranking')

    parser.add_argument('indicators', nargs='*', help='one or several yaml configuration files (default to all yaml present in current directory)')
    parser.add_argument('--areas', nargs='*', help='by default: use area field from yaml config')
    parser.add_argument('-o', '--output', default='dist', help='%(default)s')
    parser.add_argument('--no-figure', action='store_false', default=None, dest='makefig', help='do not make figures (if enabled by default)')
    parser.add_argument('--no-markdown', action='store_false', dest='markdown', help='straight to the build')

    parser.add_argument('--png', action='store_true', help='store interactive figs to png as well (for markdown rendering)')
    parser.add_argument('--build', '--html', action='store_true', help='build as HTML')
    parser.add_argument('--pdf', action='store_true', help='make pdf version when building')

    parser.add_argument('--templates-dir', default='templates', help='templates directory (default: %(default)s)')
    parser.add_argument('--skip-error', action='store_true', help='skip area with error instead of raising exception')
    parser.add_argument('--no-update', action='store_false', default=True, dest='update', help='do not update existing markdown')
    parser.add_argument('--use-time', action='store_true')
    parser.add_argument('--deploy', action='store_true', help='deploy to local isipedia.org')
    parser.add_argument('--deploy-test', action='store_true')
    parser.add_argument('--deploy-demo', action='store_true')
    parser.add_argument('--delete-rsync', action='store_true')

    parser.add_argument('--dev', action='store_true', help='development mode')
    parser.add_argument('--setup-only', action='store_true', help='only setup stage, do not actually load templates')
    # parser.add_argument('--deploy-demo', action='store_true')
    # parser.add_argument('--deploy-isipedia', action='store_true')

    # deprecated arguments, kept for back-compatibility

    o = parser.parse_args()

    # make sure that indicators are loaded from the same directory
    if o.indicators and len(set(os.path.dirname(i) for i in o.indicators)) > 1:
        parser.error('indicators must all be in one directory')

    # just pick all yaml files present in current directory if no indicator is provided
    if not o.indicators:
        o.indicators = [f for f in os.listdir() if f.endswith('.yml')]

    if os.path.exists('custom.py'):
        print('custom.py module present in work directory. Load it.')
        try:
            import custom
        except ImportError as error:
            raise

    print('country_data:', country_data_folder)

    all_md_files = []

    for indicator in o.indicators:

        if indicator.endswith('.yml'):
            indicator, _ = os.path.splitext(indicator)

        study = StudyConfig.load(indicator,
            root=o.output, templates_dir=o.templates_dir)

        if study.get('skip'):
            print('Skip', indicator)
            continue

        if not o.ranking:
            setattr(study, 'ranking-files', [])

        if not o.areas:
            o.areas = study.area or ['']

        print('#### process', indicator, {
            'makefig':o.makefig,
            'ranking': bool(study.get('ranking-files')),
            'output':study.root,
            'templates':study.templates_dir
            }, o.areas if len(o.areas) < 3 else f"{len(o.areas)} areas")

        # Write the study metadata to dist
        study_file = os.path.join(study.folder, 'metadata.json')
        print('write to', study_file)
        os.makedirs(study.folder, exist_ok=True)
        with open(study_file, 'w') as f:
            json.dump(vars(study), f, default=str)


        if o.markdown:
            try:
                md_files = process_study(study, country_names=o.areas, fail_on_error=not o.skip_error,
                    makefig=o.makefig, dev=o.dev, setup_only=o.setup_only, png=o.png, update=o.update)

                all_md_files.extend(md_files)
            except Exception as error:
                raise
                print(error)
                continue

        else:
            md_files = [TemplateContext(study, area=area).markdown for area in o.areas]

        # copy all assets
        if os.path.exists('assets'):
            cmd = ['rsync','-avzr', 'assets', study.folder+ '/']
            subprocess.check_call(cmd)


        if o.build:
            from isipedia.web import root
            import sys
            cmd = [sys.executable,os.path.join(root, 'scripts', 'process_articles.py'), '--update','--out', o.output, '--html'] + md_files
            if o.use_time: cmd += ['--use-time']
            if o.png: cmd += ['--png']
            if o.pdf: cmd += ['--pdf']
            print(' '.join(cmd))
            subprocess.run(cmd)


        def deploy(root):
            cmd = ['rsync','-avzr', os.path.join(o.output, study.url)+'/', f'{root}/{study.url}/']
            if o.delete_rsync:
                cmd.append('--delete')
            print(' '.join(cmd))
            subprocess.run(cmd)

            # cmd = ['rsync','-avzr', o.output+'/pdf/', f'{root}/pdf/']
            # # cmd = ['rsync','-avzr', os.path.join(o.output, study.url_pdf), os.path.join(root, 'dist', study.url_pdf)]
            # # cmd = ['cp', os.path.join(o.output, study.url).replace('report', 'pdf')+'*', os.path.join(root, 'dist', 'pdf')]
            # print(' '.join(cmd))
            # subprocess.run(cmd)

        def deploy_remote(site):
            import socket
            host = socket.gethostname()
            if host.startswith('login'):
                server = 'se59:'
            else:
                server = 'isipedia.org:'
            remote = server+'/webservice/'+site
            deploy(remote)


        if o.deploy:
            from isipedia.web import root
            deploy(root / 'dist')

        if o.deploy_test:
            deploy_remote('test.isipedia.org')

        if o.deploy_demo:
            deploy_remote('demo.isipedia.org')


if __name__ == '__main__':
    main()
