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


def _hashsum(string, l=6):
    return hashlib.sha1(string.encode()).hexdigest()[:l] 


def figcode(data, **kwargs):
    "code based on file name and figure arguments"
    kwargs['filename'] = data.filename
    string = json.dumps(kwargs, sort_keys=True)
    return _hashsum(string)
    # return os.path.splitext(data['filename'])[0].lower()


class SuperFig:
    backends = ['mpl', 'mpld3']
    make = None

    def __init__(self, backend, makefig=True, backends=None):
        self.backend = backend
        self.makefig = makefig
        if backends:
            self.backends = backends


    def __call__(self, data, **kwargs):
        if self.makefig:
            # self.make_figs(data, kwargs, [self.backend])
            self.make_figs(data, kwargs)
        return self.insert_cmd(data, kwargs)


    def figpath(self, data, kwargs, backend=None, relative=False):
        ext = {'mpl': 'png', 'mpld3': 'json'}[backend or self.backend]
        return os.path.join('' if relative else os.path.dirname(data.filename), 'figures', figcode(data, **kwargs) + '.' + ext)


    def caption(self, data, caption="", **kwargs):
        return caption


    def insert_cmd(self, data, kwargs, backend=None):
        backend = backend or self.backend
        if backend == 'mpl':
            cmd = '![{}]({})'.format(self.caption(data, **kwargs), self.figpath(data, kwargs, backend, relative=True))
        elif backend == 'mpld3':
            cmd = '<div id={}></div>'.format(self.figcode(data, kwargs))
        else:
            cmd = '!! Error when inserting figure: {} for backend {}'.format(self.figcode(data, kwargs), backend)
        return cmd


    def make_figs(self, data, kwargs, backends=None):
        "custom make figs that reuse same figure"
        backends = backends or self.backends
        if not backends : return
        import matplotlib.pyplot as plt
        fig, axes = self.make(data, **kwargs)
        for backend in backends:        
            if backend == 'mpl':
                fpath = _maybe_createdir(self.figpath(data, kwargs, 'mpl'))
                fig.savefig(fpath, dpi=100)
            elif backend == 'mpld3':
                import mpld3
                js = mpld3.fig_to_dict(fig)
                fpath = _maybe_createdir(self.figpath(data, kwargs, 'mpld3'))
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



def _rankingmap(ranking, x, scenario=None, maskfile='countrymasks.nc', method='number', title='', label=''):
    """
    ranking: Ranking instance
    method: "number" (default) or "value"
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import netCDF4 as nc

    if method not in ['number', 'value']:
        raise ValueError('method must be "number" or "value"')

    areas = ranking.areas

    with nc.Dataset(maskfile) as ds:
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



class LinePlot(SuperFig):
    def make(self, *args, **kwargs):
        return _lineplot(*args, **kwargs)


class RankingMap(SuperFig):
    make = _rankingmap

    def __init__(self, ranking, backend, makefig=True, backends=None):
        super().__init__(backend, makefig, backends)
        self.ranking = ranking

    def __call__(self, variable, x, **kwargs):
        return super().__call__(self.ranking[variable], x=x, **kwargs)

    def make(self, *args, **kwargs):
        return _rankingmap(*args, **kwargs)
