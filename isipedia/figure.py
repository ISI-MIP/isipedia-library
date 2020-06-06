import os
import json
import hashlib
import numpy as np
import logging

import pandas as pd
import altair as alt
import matplotlib as mpl
import matplotlib.font_manager as font_manager

from isipedia.web import isipedia_org
from isipedia.country import country_data_folder, countrymasks_folder
from isipedia.command import figures_register, isipediafigure


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        else:
            return super(NpEncoder).default(obj)


def _hashsum(kwargs, l=6):
    string = json.dumps(kwargs, sort_keys=True)    
    return hashlib.sha1(string.encode()).hexdigest()[:l] 


def _shortname(fname, area):
    ' determine variable name from json file name '
    name, ext = os.path.splitext(os.path.basename(fname))
    if name.endswith(area):
        name = name[:-len(area)-1]
    return name.replace('-','_')


def _get_json_file(context, arg):
    from isipedia.jsonfile import JsonFile
    if isinstance(arg, JsonFile):
        return arg

    if arg in context.variables:
        return context.variables[arg]

    fname = _shortname(arg, context.area)
    fnames = [_shortname(v.filename, v.area) for v in context.variables]
    if fname in fnames:
        return context.variables[fnames.index(fname)]

    fname = os.path.join(context.folder, arg)
    if os.path.exists(fname):
        if os.path.splitext(fname)[1] == '.json':
            return json.load()
        else:
            raise NotImplementedError('cannot load {}'.format(fname))

    raise ValueError('no matching variable found: '+repr(arg))


class SuperFig:
    backend = None
    prefix = ''
    ext = '.png' # static extension

    def __init__(self, context):
        self.context = context
        self.makefig = context.makefig
        self.png = context.png

    def figcode(self, *args, **kwargs):
        "code based on file name and figure arguments"
        kwargs['args'] = args
        return self.prefix + _hashsum(kwargs)

    def figpath(self, figid, relative=False):
        return os.path.join('' if relative else self.context.folder, 'figures', figid+'-'+self.backend +self.ext)

    def insert_cmd(self, figid, caption='', crossref=False):
        return ('![{}]({}){{{}}}' if crossref else '![{}]({})').format(caption, self.figpath(figid, relative=True), '#fig:'+figid)

    def caption(self, *args, **kwargs):
        return 'No Caption.'

    def __call__(self, *args, **kwargs):
        # extract markdown parameters
        caption = kwargs.pop('caption', None)
        figid = kwargs.pop('id', '')
        crossref = kwargs.pop('crossref', True)
        assert type(figid) is str, 'id parameter must be a string'
        assert caption is None or type(caption) is str, 'caption parameter must be a string'
        if not figid:
            figid = self.figcode(*args, **kwargs)
        if caption is None:
            caption = self.caption(*args, **kwargs)

        if self.makefig:
            fig = self.make(*args, **kwargs)
            figpath = self.figpath(figid)
            figdir = os.path.join(self.context.folder, 'figures')
            if not os.path.exists(figdir):
                os.makedirs(figdir)
            path_noext, ext = os.path.splitext(figpath)
            self.save_and_close(fig, path_noext) 

        return self.insert_cmd(figid, caption)

    def make(self):
        raise NotImplementedError()

    def _get_json_file(self, variable):
        return _get_json_file(self.context, variable)

    def save_and_close(self, fig, path_noext):
        if self.backend == 'mpl':
            import matplotlib.pyplot as plt
            print('saving mpl...')
            fig.savefig(path_noext+self.ext, dpi=100)
            plt.close(fig)

        elif self.backend == 'mpld3':
            import mpld3
            import matplotlib.pyplot as plt
            fig.savefig(path_noext+self.ext, dpi=100)
            js = mpld3.fig_to_dict(fig)
            fpath = path_noext+'.json'
            json.dump(js, open(fpath, 'w'), cls=NpEncoder)
            plt.close(fig)

        elif self.backend == 'vl':
            print('{}: saving json...'.format(type(self)))
            fig.save(path_noext+'.json') # json
            if self.png:
                print('{}: saving png...'.format(type(self)))
                # fig.save(path_noext+self.ext) # static
                fig.save(path_noext+'.png', scale_factor=2) # static



def _maybe_createdir(path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return path



def isipedia_theme():
    font = "IBM Plex Sans"

    return {
        "config" : {
            "background": '#F1F4F4',
            "padding": 0,
            "view": {
                "continuousWidth": 600, # this has no effect with autosize fit-x
                "continuousHeight": 300, 
                "strokeOpacity": 0, # do not show axis frame
                },
            "autosize": {"contains": "padding", "type": "fit-x"}, # this cancels continuousWidth

            "title": {
                "font": font,
                "fontsize": 16,
            },
            "text": {
                "font": font,
                "fontsize": 14,
            },
            "header": {
                "font": font,
                "titleFont": font,
                "titleFontSize": 20,
                "labelFont": font,
                "labelFontSize": 18,
            },
            "axis": {
                "labelFont": font,
                "titleFont": font,
                "labelFontSize": 14,
                "titleFontSize": 16,
            },
            "mark": {
                "font": font,
                "fontSize": 14,
            },
            "legend": {
                "labelFont": font,
                "titleFont": font,
                "labelFontSize": 14,
                "titleFontSize": 14,
            },
        }
    }

alt.themes.register('isipedia_theme', isipedia_theme)
alt.themes.enable('isipedia_theme')


# matplotlib fonts
font_dirs = [os.path.join(isipedia_org, 'assets', 'fonts')]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
font_list = font_manager.createFontList(font_files)
font_manager.fontManager.ttflist.extend(font_list)

mpl.rcParams['font.family'] = 'IBM Plex Sans'


my_dpi = 96

mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['axes.labelsize'] = 16
# mpl.rcParams['axes.labelweight'] = "bold"
# mpl.rcParams['lines.linewidth'] : 3
# mpl.rcParams['lines.markersize'] : 10
mpl.rcParams['xtick.labelsize'] = 14
# mpl.rcParams['xtick.labelweight'] = "bold"
# mpl.rcParams['ytick.labelweight'] = "bold"
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['figure.titlesize'] = 20
mpl.rcParams['figure.dpi'] = my_dpi
scale = 2
mpl.rcParams['figure.figsize'] = 600/my_dpi*scale, 300/my_dpi*scale


def get_ranking_data(countries, ranking, x, scenario=None, method='number'):
    """get ranking data for figures"""
    import pandas as pd

    if method not in ['number', 'value']:
        raise ValueError('method must be "number" or "value"')

    ranking_method = getattr(ranking, method)
    ranking_data = []
    for c in countries:
        area = c['properties']['ISIPEDIA']
        name = c['properties']['NAME']
        # print(area)
        if area.lower() not in ranking.areas:
            logging.warning('missing area for ranking: '+area)
            continue
        value = ranking.value(area.lower(), x, scenario)
        if value is not None:
            value = round(value, 2)
        rank = ranking.number(area.lower(), x, scenario)
        ranking_data.append((area, name, value, rank, ranking.plot_label_y, ranking.plot_unit_y))

    return pd.DataFrame(ranking_data, columns=["Code", "Country", "Value", "Rank", 'label', 'unit'])