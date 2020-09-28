

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



@isipediafigure(name='lineplot_mpl')
class LinePlotMpl(SuperFig):

    backend = 'mpl'

    def make(self, vname, *args, **kwargs):
        data = self._get_json_file(vname)
        fig, ax = _lineplot(data, *args, **kwargs)
        return fig

    def figcode(self, vname, *args, **kwargs):
        data = self._get_json_file(vname)
        kwargs['filename'] = data.filename
        kwargs['args'] = args
        return _hashsum(kwargs)

    def caption(self, vname, *args, **kwargs):
        data = self._get_json_file(vname)
        return data.plot_title


scenario_map = {
    'historical':'Historical',
    'rcp26':'RCP 2.6',
    'rcp60':'RCP 6',
    # 'rcp26':'Low emissions',
    # 'rcp60':'Buisness as usual',
    # 'rcp26':'Low emission scenario (RCP 2.6)',
    # 'rcp60':'Buisness as usual (RCP 6.0)',
}

def _lineplot_altair_time(data, x=None, scenario=None, climate=None, impact=None, shading=False, title='', xlabel='', ylabel=''):

    df0 = data.to_pandas()

    df0['1981-2000'].loc['rcp60'] = df0['1981-2000'].loc['historical'].values
    df0['1981-2000'].loc['rcp26'] = df0['1981-2000'].loc['historical'].values

    d = []
    for scenario in df0.index.levels[0]:
        df = df0.loc[scenario]
        lower = df.min(axis=0) / 100 # % format
        upper = df.max(axis=0) / 100
        median = df.loc['median', 'median']  / 100
        df2 = pd.DataFrame({'lower':lower, 'upper':upper, 'median':median, 'scenario': scenario})
        d.append(df2.loc['1901-1920':'2081:2100'])
        
    df2 = pd.concat(d, axis=0)
    # df2.loc['1981-2000'] = 
    df2 = df2.reset_index()

    x = [xx.split('-') for xx in df2['x']]
    df2['x'] = [(int(y1)+int(y2)-1)//2 for y1, y2 in x]


    df2 = df2.replace(scenario_map)

    # ----------------

    # color = alt.Color('scenario', scale=alt.Scale(scheme="tableau10"))  # check later
    color = alt.Color('scenario', title='Climate Scenario',
                      # scale=alt.Scale(scheme='tableau10'))
                      scale=alt.Scale(domain=list(scenario_map.values()), 
                                      range=['#4674b9','#80b946','orange']))

    # axisX = alt.X('x:Q', title=xlabel or 'Time', scale=alt.Scale(domain=[1900, 2100]))
    axisX = alt.X('x:Q', title=xlabel or 'Time', scale=alt.Scale(domain=[1900, 2100]), axis=alt.Axis(format="i", values=np.arange(1900, 2100+1, 20).tolist()))
    axisY = alt.Y('median:Q', title=data.plot_unit_y or ylabel, axis=alt.Axis(format='%'))
    
    base = alt.Chart(df2)

    lines = base.mark_line().encode(
        x = axisX,
        y = alt.Y('median:Q', title=data.plot_unit_y or ylabel),
    #     color = alt.Y('scenario:O'),
        color = color,
    )
    points = base.mark_point(size=60).encode(
        x = axisX,
        y = axisY,
        color = color,
        tooltip=[
                 alt.Tooltip('scenario:O'), 
                 alt.Tooltip('x:Q', title=xlabel or 'Time'), 
                 alt.Tooltip('median:Q', format='.1%'), 
                 alt.Tooltip('lower:Q', format='.1%'), 
                 alt.Tooltip('upper:Q', format='.1%')],
    )
    areas = base.mark_area(opacity=0.2).encode(
        x = axisX,
        y = alt.Y('lower:Q'),
        y2 = alt.Y2('upper:Q'),
        color = color,
    )

    rule_data = pd.DataFrame({'line': [2005]})
    rule_text_data = pd.DataFrame([
        {"year": 1910, "text": "Historical Period"},
        {"year": 2015, "text": "Future Projections"},
    ])
        

    rule = alt.Chart(rule_data).mark_rule().encode(
        x='line:Q'
    )

    rule_text = alt.Chart(rule_text_data).mark_text(align="left", dy=-130).encode(
        x="year",
        text="text"
    )

    # chart = points + lines + areas
    chart = (points + lines + areas + rule + rule_text).properties(
        title=title or data.plot_title,
        width=800,
        autosize=alt.AutoSizeParams(contains="padding", type="fit-x"),
    )

    return configure_chart(chart).interactive()


def _lineplot_altair_time_advanced(data, x=None, scenario=None, climate=None, impact=None, shading=False, title='', xlabel='', ylabel=''):
    import pandas as pd
    import altair as alt

    df = pd.concat([pd.DataFrame(l) for l in data.filter_lines()])

    # Divide by 100 so we can percent-format in the plot
    df.y = df.y / 100

    df["x_range"] = df.x
    df.x = df.x.apply(lambda x: int(x.split("-")[1]) - 10)
    df = df[df.x < 2100]

    # Fill in gap by duplicating historical values to future scenarios
    extra = df[(df.scenario == "historical") & (df.x == 1990)].copy()
    extra.at[:, "scenario"] = "rcp60"
    df = df.append(extra)
    extra = df[(df.scenario == "historical") & (df.x == 1990)].copy()
    extra.at[:, "scenario"] = "rcp26"
    df = df.append(extra)

    df["model"] = df.climate + " / " + df.impact

    df = df.replace(scenario_map)
    # print(df)
    # ------------------
    axisX = alt.X('x:Q', title=xlabel or 'Time', scale=alt.Scale(domain=[1900, 2100]), axis=alt.Axis(format="i", values=np.arange(1900, 2100+1, 20).tolist()))

    selection_climate = alt.selection_multi(fields=['scenario'], bind='legend')

    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=["x", 'y'], empty='none')

    base = alt.Chart(df[(df.climate != "median")])

    # color = alt.Color('scenario', scale=alt.Scale(scheme="tableau10"))
    color = alt.Color('scenario', title='Climate Scenario',
                      # scale=alt.Scale(scheme='tableau10'))
                      scale=alt.Scale(domain=list(scenario_map.values()), 
                      # scale=alt.Scale(domain=list(scenario_map.keys()), 
                                      range=['#4674b9','#80b946','orange']))    

    rule_data = pd.DataFrame({'line': [2005]})
    rule_text_data = pd.DataFrame([
        {"year": 1910, "text": "Historical Period"},
        {"year": 2015, "text": "Future Projections"},
    ])

    rule = alt.Chart(rule_data).mark_rule().encode(
        x='line:Q'
    )

    rule_text = alt.Chart(rule_text_data).mark_text(align="left", dy=-130).encode(
        x="year",
        text="text"
    )

    area = base.mark_area(opacity=0.3).encode(
        x=axisX,
        color=color,
        y=alt.Y(field="y", type="quantitative", axis=alt.Axis(format='%'), aggregate="min"),
        y2=alt.Y2(field="y", aggregate="max"),
        opacity=alt.condition(selection_climate, alt.value(0.3), alt.value(0)),
    ).add_selection(selection_climate)

    lines = base.mark_line().encode(
        x=axisX,
        y=alt.Y("y", axis=alt.Axis(format='%')),
        detail=["climate", "impact", "scenario"],
        color=color,
        opacity=alt.condition(selection_climate, alt.value(0.3), alt.value(0)), 
        size=alt.condition("datum.impact == 'median'", alt.value(5), alt.value(1))
    )

    points = base.mark_point().encode(
        x=axisX,
        y=alt.Y("y", axis=alt.Axis(title=data.plot_unit_y, format='%')),
        detail=["climate", "impact", "scenario"],
        color=color,
        opacity=alt.condition(selection_climate, alt.value(0.3), alt.value(0)), 
        size=alt.value(12),
    )

    text_model = points.mark_text(align='left', dx=-5, dy=-6).encode(
        text=alt.condition(nearest, "model", alt.value(' ')),
        opacity=alt.condition(selection_climate, alt.value(1), alt.value(0)),
        color=alt.value("black")
    ).add_selection(nearest)

    text_pct = points.mark_text(align='left', dx=-5, dy=6).encode(
        text=alt.condition(nearest, "y", alt.value(' '), format=".2p"),
        opacity=alt.condition(selection_climate, alt.value(1), alt.value(0)),
        color=alt.value("black")
    )


    chart = (area + rule + rule_text +  lines + points + text_model + text_pct).properties(
        title=title or data.plot_title,
        width=800,
        autosize=alt.AutoSizeParams(contains="padding", type="fit-x"),
    )

    # chart.save("chart.json")
    return configure_chart(chart).interactive()



def _lineplot_altair_temp(data, x=None, scenario=None, climate=None, impact=None, shading=False, title='', xlabel='', ylabel=''):
    
    # median data
    df = data.to_pandas().loc[scenario]
    lower = df.min(axis=0) / 100
    upper = df.max(axis=0) / 100
    median = df.loc['median', 'median'] / 100
    df2 = pd.DataFrame({'lower':lower, 'upper':upper, 'median':median, 'climate':'Median'}).reset_index() 

    if not title:
        title = data.plot_title
    if not xlabel: 
        xlabel = '{} ({})'.format(data.plot_label_x, data.plot_unit_x)
        # xlabel = data.plot_unit_x
    if not ylabel: 
        # ylabel = '{} ({})'.format(data.plot_label_y, data.plot_unit_y)
        ylabel = data.plot_unit_y

    # if data.plot_type == 'indicator_vs_timeslices':
    #     x = [xx.split('-') for xx in df2['x']]
    #     df2['x'] = [(int(y1)+int(y2))/2 for y1, y2 in x]
    #     axisX = alt.X('x:Q', title=xlabel, scale=alt.Scale(domain=[1900, 2100]))
    # else:
    axisX = alt.X('x:Q', title=xlabel, scale=alt.Scale(domain=[0, df2['x'].max()]), axis=alt.Axis(values=data.x))

    base = alt.Chart(df2)

    nearest = alt.selection(type='single', nearest=True, on='mouseover', empty='none')
    #                             fields=["x", "median"], empty='none')

    # axisY = alt.Y('median:Q', title=ylabel, axis=alt.Y(format='%'))
    axisY = alt.Y('median:Q', title=ylabel, axis=alt.Axis(format='%'))

    color = 'orange'
    color2 = alt.Color('climate', title='Climate Model',
                      # scale=alt.Scale(scheme='tableau10'))
                      scale=alt.Scale(domain=['Median'], 
                                      range=[color]))

    area = base.mark_area(opacity=0.3, color=color).encode(
        x=axisX,
        y=alt.Y('lower:Q'),
        y2=alt.Y2('upper:Q'),
    )

    lines = base.mark_line(color=color).encode(
        x=axisX,
        y=axisY,
    )

    points = base.mark_point(size=60).encode(
        x=axisX,
        y=axisY,
        color=color2,
        tooltip=[alt.Tooltip('x:Q', title=xlabel), alt.Tooltip('median:Q', title=ylabel, format='.1%')],
    )

    chart = (points + lines + area).properties(
            title=title,
            width=800,
            autosize=alt.AutoSizeParams(contains="padding", type="fit-x"),
        )

    return configure_chart(chart).interactive()



@isipediafigure(name='lineplot')
class LinePlot(SuperFig):

    backend = 'vl'

    def make(self, vname, *args, **kwargs):
        data = self._get_json_file(vname)
        if data.plot_type == 'indicator_vs_temperature':
            return _lineplot_altair_temp(data, *args, **kwargs)
        else:
            return _lineplot_altair_time(data, *args, **kwargs)

    def figcode(self, vname, *args, **kwargs):
        data = self._get_json_file(vname)
        kwargs['filename'] = data.filename
        kwargs['args'] = args
        return _hashsum(kwargs)

    def caption(self, vname, *args, **kwargs):
        data = self._get_json_file(vname)
        return data.plot_title



def _lineplot_altair_temp_advanced(data, x=None, scenario=None, climate=None, impact=None, shading=False, title='', xlabel='', ylabel=''):
    import pandas as pd
    import altair as alt

    df = pd.concat([pd.DataFrame(l) for l in data.filter_lines()])

    df["model"] = df.climate + " / " + df.impact

    # Divide by 100 so we can percent-format in the plot
    df.y = df.y / 100

    selection_climate = alt.selection_multi(fields=['climate'], bind='legend')

    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=["x", 'y'], empty='none')

    base = alt.Chart(df[(df.climate != "median")])

    color = alt.Color('climate', scale=alt.Scale(scheme="dark2"), title='Climate Model')

    area = base.mark_area(opacity=0.3).encode(
        x=alt.X("x", scale=alt.Scale(domain=[0, df['x'].max()])),
        color=color,
        y=alt.Y(field="y", type="quantitative", axis=alt.Axis(format='%'), aggregate="min"),
        y2=alt.Y2(field="y", aggregate="max"),
        opacity=alt.condition(selection_climate, alt.value(0.3), alt.value(0)),
    ).add_selection(selection_climate)

    lines = base.mark_line().encode(
        x=alt.X("x"),
        y=alt.Y("y", axis=alt.Axis(format='%')),
        detail=["climate", "impact"],
        color=color,
        opacity=alt.condition(selection_climate, alt.value(0.3), alt.value(0)), 
        size=alt.condition("datum.impact == 'median'", alt.value(5), alt.value(1)),
    )

    points = base.mark_point().encode(
        x=alt.X("x", axis=alt.Axis(title=xlabel or '{} ({})'.format(data.plot_label_x, data.plot_unit_x), values=data.x)),
        y=alt.Y("y", axis=alt.Axis(title=data.plot_unit_y, format='%')),
        detail=["climate", "impact"],
        color=color,
        opacity=alt.condition(selection_climate, alt.value(0.3), alt.value(0)), 
        size=alt.value(12)
    )

    text_model = points.mark_text(align='left', dx=-5, dy=-6).encode(
        text=alt.condition(nearest, "model", alt.value(' ')),
        opacity=alt.condition(selection_climate, alt.value(1), alt.value(0)),
        color=alt.value("black")
    ).add_selection(nearest)

    text_pct = points.mark_text(align='left', dx=-5, dy=6).encode(
        text=alt.condition(nearest, "y", alt.value(' '), format=".2p"),
        opacity=alt.condition(selection_climate, alt.value(1), alt.value(0)),
        color=alt.value("black")
    )

    chart = (area + lines + points + text_model + text_pct ).properties(
        title=data.plot_title,
        width=800,
        autosize=alt.AutoSizeParams(contains="padding", type="fit-x"),
    )

    return configure_chart(chart).interactive()


@isipediafigure(name='lineplot_advanced')
class LinePlot2(LinePlot):

    def make(self, vname, *args, **kwargs):
        data = self._get_json_file(vname)
        if data.plot_type == 'indicator_vs_temperature':
            return _lineplot_altair_temp_advanced(data, *args, **kwargs)
        else:
            return _lineplot_altair_time_advanced(data, *args, **kwargs)


def _rankingmap_mpl(countrymasksnc, ranking, x, scenario=None, method='number', title='', label=''):
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

    return fig


def _rankingmap_altair(countries, ranking, x, scenario=None, method='number', title='', label=''):
    # Adapted from https://altair-viz.github.io/gallery/index.html

    import pandas as pd
    import altair as alt

    if method not in ['number', 'value']:
        raise ValueError('method must be "number" or "value"')

    source = alt.Data(values=countries)

    if ranking.plot_type == 'indicator_vs_temperature':
        details = 'warming level: {} {}'.format(x, ranking.plot_unit_x)
    else:
        details = 'period: {}, scenario: {}'.format(x, {'rcp26':'RCP 2.6', 'rcp45':'RCP 4.5', 'rcp60':'RCP 6', 'rcp85':'RCP 8.5'}.get(scenario, scenario))
    default_title = getattr(ranking, 'plot_label_y','') + '\n' + details
    # default_label = 'ranking number' if method == 'number' else ('ranking value ({})'.format(getattr(ranking, 'plot_unit_y')))    

    ranking_data = get_ranking_data(countries, ranking, x, scenario, method)

    chart = alt.Chart(source).mark_geoshape().encode(
        # color="Rank:Q",
        color=alt.Color("Rank:Q", sort='ascending') if method == 'number'  else alt.Color("Value:Q", sort='descending'),
        # tooltip=["Country:N", "Code:N", "Value:Q", "Rank:Q"]
        tooltip=["label:N", "unit:N", "Country:N", "Code:N", "Value:Q", "Rank:Q"]
    ).transform_lookup(
        lookup='properties.ISIPEDIA',
        from_=alt.LookupData(ranking_data, 'Code', ranking_data.columns.tolist())
    ).project(
        'naturalEarth1'
    ).properties(width=800, autosize=alt.AutoSizeParams(contains="padding", type="fit-x"), title=ranking.plot_title
    # ).configure_view(stroke=None
    ).configure(background='#F1F4F4'
    ).configure_title(
        fontSize=16,
    ).configure_axis(
        labelFontSize=14,
        titleFontSize=16,
    ).configure_legend(
        titleFontSize=14,
        labelFontSize=14,
    ).configure_mark(
        fontSize=14
    )
    # ).interactive()


    return chart


@isipediafigure(name='rankingmap')
class RankingMap(SuperFig):
    backend = 'vl'

    def make(self, variable, *args, **kwargs):
        # return _rankingmap_mpl(self.context.countrymasksnc, self.context.ranking[variable.replace('-','_')], *args, **kwargs)
        return _rankingmap_altair(self.context.countries_simple, self.context.ranking[variable.replace('-','_')], *args, **kwargs)

    def figcode(self, variable, x, **kwargs):
        kwargs['x'] = x
        kwargs['variable'] = variable   # string
        return 'rankingmap-'+_hashsum(kwargs)

    def caption(self, variable, *args, **kwargs):
        ranking = self.context.ranking[variable.replace('-','_')]
        return ranking.plot_title


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
    def __init__(self, indicator, studytype, cube_path):
        self.indicator = indicator
        self.studytype = studytype
        self.cube_path = cube_path
        self._data = {}
        self._areas = {}


    def csvpath(self, name, x, scenario=None, climate=None, impact=None):
        mapfile = '{scenario}_{climate}_{impact}_{x}.csv'.format(
            scenario=scenario or 'all', 
            climate=climate or 'median',
            impact=impact or 'median',
            x='{:.1f}'.format(x) if type(x) is not str else x)
        return os.path.join('maps', self.indicator, name, mapfile)


    def loadcsv(self, name, x, scenario=None, climate=None, impact=None, fill_value=0):
        mpath = self.csvpath(name, x, scenario, climate, impact)
        x = np.genfromtxt(mpath, delimiter=',')
        if fill_value is not None:
            x[np.isnan(x)] = fill_value
        return x


    def bounds(self, area):
        if area not in self._areas:
            boundspath = os.path.join(country_data_folder, area, 'bounds.json')
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

    return fig


def _countrymap_altair(mapdata, countrymasksnc, jsfile, x=None, scenario=None, climate=None, impact=None, title='', label=''):
    """
    """
    import altair as alt
    import numpy as np
    import pandas as pd

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

    localmap = localmap[::-1]
    mask = mask[::-1]

    ni, nj = localmap.shape
    # l, r, b, t = bnds.extent
    # x = np.linspace(l, r, nj)
    # y = np.linspace(t, b, ni)
    l, r, b, t = bnds.indices
    x = countrymasksnc['lon'][l:r+1]
    y = countrymasksnc['lat'][t:b+1][::-1]
    X, Y = np.meshgrid(x, y)

    l, r, b, t = bnds.extent
    aspect = (t - b) / (r - l) / np.cos(np.deg2rad((t+b)/2))
    print(jsfile.area, 'aspect', aspect)


    # Convert this grid to columnar data expected by Altair
    source = pd.DataFrame({'lon': X[mask].round(2),
                         'lat': Y[mask].round(2),
                         'z': localmap[mask]})


    chart = alt.Chart(source).mark_rect().encode(
        x='lon:O',
        y=alt.Y('lat:O', sort='descending'),
        color=alt.Color('z:Q', title=''),
        tooltip=[alt.Tooltip('z:Q', title='{} ({})'.format(jsfile.plot_label_y, jsfile.plot_unit_y)), 'lon:Q', 'lat:Q']
    ).properties(title=jsfile.plot_title,
    width=800, height=int(800*aspect), autosize=alt.AutoSizeParams(contains="padding", type="fit-x"), 
    ).configure(
    background='#F1F4F4'
    ).configure_header(
    titleFont="IBM Plex Sans",
    titleFontSize=20,
    labelFont="IBM Plex Sans",
    labelFontSize=18,    
    ).configure_title(
        fontSize=16,
    ).configure_axis(
        labelFontSize=14,
        titleFontSize=16,
    ).configure_legend(
        titleFontSize=14,
        labelFontSize=14,
    ).configure_mark(
        fontSize=14
    ).interactive()

    return chart
    # # add country borders
    # source = alt.Data(values=[c for c in countries if c['properties']['ISIPEDIA']==area])
    # # source = alt.Data(values=[c for c in js['features']])
    # borders = alt.Chart(source).mark_geoshape(stroke='black', fill='none', fillOpacity=0)
    # borders


    # return chart + borders



@isipediafigure(name='countrymap')
class CountryMap(SuperFig):
    backend = 'vl'

    def make(self, vname, *args, **kwargs):
        jsfile = self._get_json_file(vname)
        # return _countrymap(self.context.mapdata, self.context.countrymasksnc, jsfile, *args, **kwargs)
        return _countrymap_altair(self.context.mapdata, self.context.countrymasksnc, jsfile, *args, **kwargs)

    def figcode(self, jsfile, x, **kwargs):
        jsfile = self._get_json_file(jsfile)
        kwargs['x'] = x
        kwargs['filename'] = jsfile.filename
        return _hashsum(kwargs)

    def caption(self, vname, *args, **kwargs):
        jsfile = self._get_json_file(vname)
        return jsfile.plot_title


@isipediafigure(name='countrymap_mpl')
class CountryMapMpl(CountryMap):
    backend = 'mpl'

    def make(self, vname, *args, **kwargs):
        jsfile = self._get_json_file(vname)
        return _countrymap(self.context.mapdata, self.context.countrymasksnc, jsfile, *args, **kwargs)
        # return _countrymap_altair(self.context.mapdata, self.context.countrymasksnc, jsfile, *args, **kwargs)
