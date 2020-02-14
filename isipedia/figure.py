import os
import json
import hashlib


import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        else:
            return super(NpEncoder).default(obj)


def _hashsum(kwargs, l=6):
    string = json.dumps(kwargs, sort_keys=True)    
    return hashlib.sha1(string.encode()).hexdigest()[:l] 


class SuperFig:
    backends = ['mpl']
    make = None

    def __init__(self, folder, backend, makefig=True, backends=None):
        self.folder = folder
        self.backend = backend
        self.makefig = makefig
        if backends:
            self.backends = backends

    def figcode(self, *args, **kwargs):
        "code based on file name and figure arguments"
        kwargs['args'] = args
        return _hashsum(kwargs)

    def __call__(self, *args, **kwargs):
        if self.makefig:
            # self.make_figs(data, kwargs, [self.backend])
            self.make_figs(args, kwargs)
        return self.insert_cmd(args, kwargs)


    def figpath(self, args, kwargs, backend=None, relative=False):
        ext = {'mpl': 'png', 'mpld3': 'json'}[backend or self.backend]
        return os.path.join('' if relative else self.folder, 'figures', self.figcode(*args, **kwargs) + '.' + ext)

    def insert_cmd(self, args, kwargs, backend=None):
        backend = backend or self.backend
        if backend == 'mpl':
            cmd = '![{}]({})'.format(kwargs.get('caption',''), self.figpath(args, kwargs, backend, relative=True))
        elif backend == 'mpld3':
            cmd = '<div id={}>{}</div>'.format(self.figcode(*args, **kwargs), kwargs.get('caption'))
        else:
            cmd = '!! Error when inserting figure: {} for backend {}'.format(self.figcode(*args, **kwargs), backend)
        return cmd


    def make_figs(self, args, kwargs, backends=None):
        "custom make figs that reuse same figure"
        backends = backends or self.backends
        if not backends : return
        import matplotlib.pyplot as plt
        caption = kwargs.pop('caption','')
        fig, axes = self.make(*args, **kwargs)
        for backend in backends:        
            if backend == 'mpl':
                fpath = _maybe_createdir(self.figpath(args, kwargs, 'mpl'))
                fig.savefig(fpath, dpi=100)
            elif backend == 'mpld3':
                import mpld3
                js = mpld3.fig_to_dict(fig)
                fpath = _maybe_createdir(self.figpath(args, kwargs, 'mpld3'))
                json.dump(js, open(fpath, 'w'), cls=NpEncoder)
        plt.close(fig)


def _maybe_createdir(path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return path



def _lineplot(data, x=None, scenario=None, climate=None, impact=None, shading=False, title='', xlabel='', ylabel=''):
    import matplotlib.pyplot as plt
    import numpy as np

    lines = data.filter_lines(scenario, climate, impact)

    if data.plot_type == 'indicator_vs_timeslices':
        # lines0 = data.filter_lines(scenario='historical', climate, impact)
        x = [(y1+y2)/2 for y1, y2 in data.timeslices_list]
        xlim = 1900, 2100
        # xlim = x[data.x.index('1901-1920')]-0.5, x[data.x.index('2081-2100')]+0.5
        xticks = np.arange(1910, 2100, 20).tolist()
    else:
        x = data.x
        xlim = x[0], x[-1]
        xticks = data.x
        # xticklabels = data.x

    fig, ax = plt.subplots(1,1)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']  
    markers = list('oxs+.<>^v*')
    styles = ['-','--',':','.-']

    # colors for scenarios
    scenarios = list(dict.fromkeys(l['scenario'] for l in lines if l['scenario'] != 'historical')) # dict maintain order in 3.7
    # linestyle for climate model
    climate_models = list(dict.fromkeys(l['climate'] for l in lines if l['climate'] != 'median')) # dict maintain order in 3.7
    # marker style for impact model
    impact_models = list(dict.fromkeys(l['impact'] for l in lines if l['impact'] != 'median')) # dict maintain order in 3.7

    for l in lines:
        # color = '#1f77b4'
        color = colors[scenarios.index(l['scenario'])] if l['scenario'] != 'historical' else 'black'
        # color = 'k'
        alpha = 0.5
        style = '-'
        marker = ''
        zorder = 1 if l['scenario'] == 'historical' else 0

        if l['climate'] == 'median' and l['impact'] == 'median':
            linewidth = 4
            alpha = 1
            # color = 'k'
        elif l['impact'] == 'median':
            continue  # dont show that one, too many curves
        elif l['climate'] == 'median':
            continue  # dont show that one, too many curves
            # linewidth = 2
            # color = '#2ca02c'
        else:
            linewidth = 0.75
            # color = colors[climate_models.index(l['climate'])]
            style = styles[climate_models.index(l['climate'])]
            marker = markers[impact_models.index(l['impact'])]

        y, lower, upper = l['y'], l['lower'], l['upper']

        # fill in last data point
        if data.plot_type == 'indicator_vs_timeslices' and l['scenario'].lower().startswith('rcp'):
            historical = data.loc('historical', l['climate'], l['impact'])
            idx = x.index((1981+2000)/2)
            y[idx] = historical['y'][idx]
            lower[idx] = historical['lower'][idx]
            upper[idx] = historical['upper'][idx]

        if shading:
            ax.fill_between(x, lower, upper, color=color, alpha=0.2)
        ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha, marker=marker, linestyle=style, zorder=zorder)


    ax.set_xticks(xticks)
    # ax.set_xticklabels(xticklabels)
    ax.set_xlim(*xlim)
    ax.set_title(title or getattr(data, 'plot_label_y',''))
    xlab = getattr(data, 'plot_label_x','')
    xunits = getattr(data, 'plot_unit_x','')
    xlab2 = '{} ({})'.format(xlab, xunits) if xunits else xlab
    ylab2 = getattr(data, 'plot_unit_y','')
    ax.set_xlabel(xlabel or xlab2)
    ax.set_ylabel(ylabel or ylab2)

    return fig, ax



def _rankingmap(countrymasksnc, ranking, x, scenario=None, method='number', title='', label=''):
    """
    countrymasksnc : nc.Dataset instance of countrymasks.nc
    ranking: Ranking instance
    method: "number" (default) or "value"
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if method not in ['number', 'value']:
        raise ValueError('method must be "number" or "value"')

    areas = ranking.areas

    ds = countrymasksnc

    lat, lon = ds['lat'][:], ds['lon'][:]
    ni, nj = lat.size, lon.size
    data = np.empty((ni, nj), dtype=int if method == 'number' else float)
    mask = np.ones((ni, nj), dtype=bool)  # undefined areas
    for area in areas:
        if 'm_'+area not in ds.variables:
            print('! rankingmap::', area, 'not found in counrty masks')
            continue

        value = getattr(ranking, method)(area, x, scenario)
        if value == 'undefined':
            print('! rankingmap::value', area, 'undefined')
            continue

        m = ds['m_'+area][:] > 0
        data[m] = value
        mask[m] = False 


    fig, ax = plt.subplots(1,1)
    h = ax.imshow(np.ma.array(data, mask=mask), extent=[-180, 180, -90, 90], 
        cmap=plt.cm.viridis_r if method == "number" else plt.cm.viridis, 
        vmax=len(areas) if method=='number' else None)

    # default_title = getattr(ranking, 'plot_label_y','')+' :: ranking: '+method
    if ranking.plot_type == 'indicator_vs_temperature':
        details = 'warming level: {} {}'.format(x, ranking.plot_unit_x)
    else:
        details = 'period: {}, scenario: {}'.format(x, {'rcp26':'RCP 2.6', 'rcp45':'RCP 4.5', 'rcp60':'RCP 6', 'rcp85':'RCP 8.5'}.get(scenario, scenario))
    default_title = getattr(ranking, 'plot_label_y','') + '\n' + details
    default_label = 'ranking number' if method == 'number' else ('ranking value ({})'.format(getattr(ranking, 'plot_unit_y')))

    ax.set_title(title or default_title)
    plt.colorbar(h, ax=ax, orientation='horizontal', label=label or default_label)    

    return fig, ax


class MapBounds:
    def __init__(self, indices, extent, splitted=False):
        self.indices = indices
        self.extent = extent
        self.splitted = splitted

    @classmethod
    def load(cls, fname):
        bounds = json.load(open(fname))
        b = bounds['indices']
        indices = b['left'], b['right'], b['bottom'], b['top']
        b = bounds['bounds']
        extent = b['left'], b['right'], b['bottom'], b['top']
        return cls(indices, extent, bounds['splitted'])

    def extract(self, x):
        l, r, b, t = self.indices
        if self.splitted:  # [0, 360]
            ni, nj = x.shape
            x = np.concatenate((x[:, -nj//2:], x[:, :nj//2]), axis=1)
        return x[t:b+1, l:r+1]


class MapData:
    """lazy loading of map data
    """
    def __init__(self, indicator, studytype, cube_path, country_data_path):
        self.indicator = indicator
        self.studytype = studytype
        self.cube_path = cube_path
        self.country_data_path = country_data_path
        self._data = {}
        self._areas = {}


    def csvpath(self, name, x, scenario=None, climate=None, impact=None):
        mapfile = '{scenario}_{climate}_{impact}_{x}.csv'.format(
            scenario=scenario or 'all', 
            climate=climate or 'median',
            impact=impact or 'median',
            x='{:.1f}'.format(x) if type(x) is not str else x)
        return os.path.join(self.cube_path, self.indicator, self.studytype, 'world' ,'maps', name, mapfile)


    def loadcsv(self, name, x, scenario=None, climate=None, impact=None, fill_value=0):
        mpath = self.csvpath(name, x, scenario, climate, impact)
        x = np.genfromtxt(mpath, delimiter=',')
        if fill_value is not None:
            x[np.isnan(x)] = fill_value
        return x


    def bounds(self, area):
        if area not in self._areas:
            boundspath = os.path.join(self.country_data_path, area, 'bounds.json')
            self._areas[area] = MapBounds.load(boundspath)
        return self._areas[area]


    def get(self, name, x, scenario=None, climate=None, impact=None):
        if name not in self._data:
            self._data[name] = self.loadcsv(name, x, scenario, climate, impact)
        return self._data[name]


def _countrymap(mapdata, countrymasksnc, jsfile, x=None, scenario=None, climate=None, impact=None, title='', label=''):
    """
    """
    import matplotlib.pyplot as plt
    import numpy as np

    area = jsfile.area

    name, ext = os.path.splitext(os.path.basename(jsfile.filename))
    if name.endswith(area):
        name = name[:-len(area)-1]

    bnds = mapdata.bounds(area)
    worldmap = mapdata.get(name, x, scenario, climate, impact)
    localmap = bnds.extract(worldmap)
    if 'm_'+area in countrymasksnc.variables:
        mask = bnds.extract(countrymasksnc['m_'+area]) > 0
    elif area == 'world':
        mask = np.zeros_like(worldmap, dtype=bool)
        for k in countrymasksnc.variables:
            if not k.startswith('m_'): 
                continue
            mask[countrymasksnc[k][:]>0] = True
    else:
        mask = np.ones_like(worldmap, dtype=bool)


    fig, ax = plt.subplots(1,1)
    if area != 'world':
        h2 = ax.imshow(localmap, extent=bnds.extent, alpha=0.5) # transparency for outside values
    h = ax.imshow(np.ma.array(localmap, mask=~mask), extent=bnds.extent)

    # default_title = getattr(ranking, 'plot_label_y','')+' :: ranking: '+method
    if jsfile.plot_type == 'indicator_vs_temperature':
        details = 'warming level: {} {}'.format(x, jsfile.plot_unit_x)
    else:
        details = 'period: {}, scenario: {}'.format(x, {'rcp26':'RCP 2.6', 'rcp45':'RCP 4.5', 'rcp60':'RCP 6', 'rcp85':'RCP 8.5'}.get(scenario, scenario))
    if climate: details += ', climate: {}'.format(climate)
    if impact: details += ', impact: {}'.format(impact)
    default_title = getattr(jsfile, 'plot_label_y','') + '\n' + details
    default_label = getattr(jsfile, 'plot_unit_y')

    ax.set_title(title or default_title)
    plt.colorbar(h, ax=ax, orientation='horizontal', label=label or default_label)

    return fig, ax


class LinePlot(SuperFig):

    def make(self, *args, **kwargs):
        return _lineplot(*args, **kwargs)

    def figcode(self, jsfile, **kwargs):
        kwargs['filename'] = jsfile.filename
        return _hashsum(kwargs)


class CountryMap(SuperFig):

    def __init__(self, mapdata, countrymasksnc, folder, backend, makefig=True, backends=None):
        self.mapdata = mapdata
        self.countrymasksnc = countrymasksnc
        super().__init__(folder, backend, makefig, backends)

    def make(self, *args, **kwargs):
        return _countrymap(self.mapdata, self.countrymasksnc, *args, **kwargs)

    def figcode(self, jsfile, x, **kwargs):
        kwargs['x'] = x
        kwargs['filename'] = jsfile.filename
        return _hashsum(kwargs)



class RankingMap(SuperFig):

    def __init__(self, ranking, countrymasksnc, folder, backend, makefig=True, backends=None):
        super().__init__(folder, backend, makefig, backends)
        self.ranking = ranking
        self.countrymasksnc = countrymasksnc


    def make(self, variable, *args, **kwargs):
        return _rankingmap(self.countrymasksnc, self.ranking[variable], *args, **kwargs)

    def figcode(self, variable, x, **kwargs):
        kwargs['x'] = x
        kwargs['variable'] = variable   # string
        return _hashsum(kwargs)
