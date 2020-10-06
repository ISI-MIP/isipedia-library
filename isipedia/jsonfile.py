"""Read nested json file from the cube
"""
import os
import json
import itertools
import copy
import logging


class File:
    def get_metadata(self):
        return {k:v for k,v in vars(self).items() if k != 'data' and not k.startswith('_')}

    # ------------------
    # back compatibility
    # ------------------
    def index(self, x):
        time_map = {
          "farfuture" : "2081-2100",
          "nearfuture" : "2041-2060",
          "today" : "2001-2020",
        }
        return self.x.index(time_map.get(x, x))

    def getarray(self, scenario=None, climate=None, impact=None, field='y'):
        return self.loc(scenario, climate, impact)[field]

    def get(self, x, scenario=None, climate=None, impact=None, field='y'):
        """get single number """
        return self.getarray(scenario, climate, impact, field)[self.index(x)]


class JsonFile(File):
    """json file close to its original structure
    """
    def __init__(self, **js):
        self._js = js
        vars(self).update(js)

        if js['plot_type'] == 'indicator_vs_temperature':
          self.x = js['temperature_list']
        else:
          self.x = ['{}-{}'.format(y1, y2) for y1, y2 in js['timeslices_list']]

        self.dims = ['scenario', 'climate', 'impact']
        self._dims = ['climate_scenario_list', 'climate_model_list', 'impact_model_list']
        self.axes = [copy.copy(js.get(d, [None])) for d in self._dims]  # [None] if vs-temperature

        # append median to the axis
        self.axes[1].append('median')
        self.axes[2].append('median')
        self.shape = [len(ax) for ax in self.axes]  # for conversion to cube if needed

    @property
    def _indices(self):
        return itertools.product(*self.axes)

    def loc(self, scenario, climate_model, impact_model):
        """leave scenario to None for vs-temp series
        """
        js = self._js
        scenario_data = js['data'][scenario] if scenario else js['data']
        climate_model_data = scenario_data[climate_model if climate_model and climate_model != 'median' else 'overall']
        if impact_model and impact_model != 'median':
            # line = climate_model_data['runs'][impact_model].copy()
            # line['y'] = line.pop('mean')  # 'y' field to unify things, copy the rest
            element = climate_model_data['runs'][impact_model]
            key = 'mean'
        else:
            element = climate_model_data
            key = 'median'

        line = {
            'x': self.x,
            'y': element[key],
            'lower': element['shading_lower_border'],
            'upper': element['shading_upper_border'],
            'scenario': scenario,
            'climate': climate_model,
            'impact': impact_model,
        }
        return line


    @property
    def lines(self):
        return [self.loc(scenario, climate, impact) for scenario, climate, impact in self._indices]


    def apply(self, func, inplace=False):
        import copy
        if inplace:
            v = self
        else:
            v = copy.deepcopy(self)
        for line in v.lines:
            for key in ['y','upper','lower']:
                y = line[key]
                for i in range(len(y)):
                    if y[i] is not None:
                        y[i] = func(y[i])
        if not inplace:
            return v


    def to_array(self, field='y'):
        import numpy as np
        cube = np.empty(tuple(self.shape) + (len(self.x), ))
        cube.fill(np.nan)
        for i, (s, sgroup) in enumerate(itertools.groupby(self.lines, key=lambda l: l['scenario'])):
            for j, (cm, cgroup) in enumerate(itertools.groupby(sgroup, key=lambda l: l['climate'])):
                for k, (im, igroup) in enumerate(itertools.groupby(cgroup, key=lambda l: l['impact'])):
                    lines_ = list(igroup)
                    assert(len(lines_) == 1)
                    l = lines_[0]
                    cube[i, j, k] = l[field]
        return cube


    def to_dimarray(self, field='y'):
        import dimarray as da
        cube = self.to_array(field)
        return da.DimArray(cube, axes=self.axes+[self.x], dims=self.dims+['x'])


    def to_pandas(self, field='y'):
        import pandas as pd
        index = pd.MultiIndex.from_tuples(self._indices, names=self.dims)
        columns = pd.Index(self.x, name='x')
        return pd.DataFrame([l[field] for l in self.lines], index=index, columns=columns)


    @classmethod
    def load(cls, fname):
        js = json.load(open(fname))
        js['filename'] = fname
        # js['id'] = file_id(fname)os.path.splitext(fname)[0].replace('-','_')  # variable name
        return cls(**js)


    def filter_indices(self, scenario=None, climate=None, impact=None):
        scenarios = self.axes[0] if scenario is None else [scenario]
        climates = self.axes[1] if climate is None else [climate]
        impacts = self.axes[2] if impact is None else [impact]
        return itertools.product(scenarios, climates, impacts)


    def filter_lines(self, scenario=None, climate=None, impact=None):
        return [self.loc(scenario, climate, impact) for scenario, climate, impact in self.filter_indices(scenario, climate, impact)]


    def getall(self, x, scenario=None, climate=None, impact=None, field='y'):
        """get an array of values for all climate and impact models
        """
        lines = self.filter_lines(scenario, climate, impact)
        index = self.index(x)
        return [l[field][index] for l in lines if l[field][index] is not None]


    def get_metadata(self):
        return {k:v for k,v in self._js.items() if k != 'data'}

    # getall = getens  # alias for backcompatibility


def _get_csv_dimension(data, candidate_keys, values=True):
    dim_list = []
    for key in candidate_keys:
        if key in data.columns:
            if values:
                dim_list = list(data[key].unique()) # assumes we have a pandas dataframe
            else:
                dim_list = None
            column = list(data.columns).index(key)
            return dim_list, column

    raise ValueError(f'dimension not found: {candidate_keys}')


class CsvFile(File):
    """CSV file in (almost) long-format, to mimic original json file

    e.g. time, scenario, climate, impact, median, lower, upper
    """
    # get, getarray, x, climate_scenario_list

    def __init__(self, data, _index_col=0, **kwargs):
        self.data = data
        vars(self).update(kwargs)

        self._index_col = _index_col
        _, self._median_col = _get_csv_dimension(data, ['median'], values=False)
        _, self._upper_col = _get_csv_dimension(data, ['upper'], values=False)
        _, self._lower_col = _get_csv_dimension(data, ['lower'], values=False)

        self.climate_model_list, self._climate_col = _get_csv_dimension(data, ['climate', 'climate_model'])

        try:
            self.climate_scenario_list, self._scenario_col = _get_csv_dimension(data, ['scenario', 'climate_scenario', 'ghg_concentration_scenario'])
        except:
            # self.climate_scenario_list = None
            pass
         # Exception as error:
         #    logging.warning(f'scenario dimension not found. Got: {data.columns}')

        self.climate_model_list, self._climate_col = _get_csv_dimension(data, ['climate', 'climate_model'])
        self.impact_model_list, self._impact_col = _get_csv_dimension(data, ['impact', 'impact_model'])

        self.x = list(self.data.iloc[:, self._index_col].unique()) # first columns

        # use time period when time...
        if self.x and self.x[0] > 1900:
            # 2010 => 2001-2020
            # 2080 => 2081-2100
            self.x = [f"{y-9}-{y+10}" for y in self.x]

    @classmethod
    def load(cls, fname, **kwargs):
        import pandas as pd
        data = pd.read_csv(fname, **kwargs)
        return cls(data, filename=fname)

    def loc(self, scenario, climate_model, impact_model):
        """leave scenario to None for vs-temp series
        """
        data = self.data
        if scenario:
            data = data[ data.iloc[:, self._scenario_col] == scenario ]
        data = data[ data.iloc[:, self._climate_col] == (climate_model or "multi-model-median") ]
        data = data[ data.iloc[:, self._impact_col] == (impact_model or "multi-model-median") ]

        line = {
            # 'x': self.x,
            'x': list(data.iloc[:, self._index_col]),
            'y': list(data.iloc[:, self._median_col]),
            'upper': list(data.iloc[:, self._upper_col]),
            'lower': list(data.iloc[:, self._lower_col]),
            'scenario': scenario,
            'climate': climate_model,
            'impact': impact_model,
        }
        # print(line)
        return line


    def apply(self, func, inplace=False):
        import copy
        if inplace:
            v = self
        else:
            v = copy.deepcopy(self)

        for j in [self._median_col, self._upper_col, self._lower_col]:
            v.data.iloc[:, j] = func(v.data.iloc[:, j])

        return v

