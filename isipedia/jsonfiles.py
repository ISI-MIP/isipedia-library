import os
import numpy as np
import netCDF4 as nc
import dimarray as da
import json

CONFIG_FILE_DEFAULT = os.path.join(os.path.dirname(__file__), 'config.json')

class Config:
    def __init__(self, configfile=CONFIG_FILE_DEFAULT, **kw):
        vars(self).update(json.load(open(configfile)))
        for k in kw:
            if not hasattr(self, k):
                raise ValueError('unknown config param: '+repr(k))
        vars(self).update(kw)

# from isipedia.config import cube_data_stefan, cube_data_out, mask_file, totpopfile, gridareafile


class Variable:
    def __init__(self, name, config=None):
        """ e.g. land-area-affected-by-drought-absolute-changes_ISIMIP-projections_versus-temperature-change/
        """
        self.name = name
        self.config = config or Config()
        vars(self).update(self._parse_name())
        assert name == self._buildname(**vars(self)), 'Expected: {}, Got: {}'.format(name, self._buildname(**vars(self)))
        self._init_axes()

    @classmethod
    def fromparams(cls, **params):
        return cls(cls._buildname(**params))

    @staticmethod
    def _buildname(**params):
        return "{exposure}-{indicator}-{change}_{studytype}_{axis}".format(**params).replace('-_','_')


    def _parse_name(self):
        variable, studytype, axis = self.name.split('_')
        prefixes = ['land-area-affected-by', 'population-exposed-to']
        prefix = ''
        for prefix in prefixes:
            if variable.startswith(prefix):
                variable = variable[len(prefix)+1:]
                break
        suffixes = ['absolute-changes', 'relative-changes']
        change = ''
        for suffix in suffixes:
            if variable.endswith(suffix):
                variable = variable[:-len(suffix)-1]
                change = suffix
                break

        return {'exposure':prefix, 'indicator':variable, 'change':change, 'studytype':studytype, 'axis':axis}
    
    @property
    def ncvariable(self):    
        params = vars(self).copy()
        if params['change']:
            return '{changeS}-in-{exposure}-{indicator}'.format(changeS=params.pop('change')[:-1], **params).replace('-','_')
        else:
            return '{exposure}-{indicator}'.format(changeS=params.pop('change')[:-1], **params).replace('-','_')

    def areanc(self, area):
        params = vars(self).copy()
        axis = params.pop('axis')
        axis = 'versus-year' if axis in ['versus-timeslices', 'versus-years'] else axis
        return (self.config.cube_netcdf +
                '{variable}/future-projections/{area}/{variable}_future-projections_{area}_{axis}.nc'.format(
                    variable=self.ncvariable.replace('_','-'), axis=axis, area=area, **params))

    @property
    def gridnc(self):
        return self.areanc('grid-level')

    def jsonfile(self, area):
        return os.path.join(self.config.cube_json, self.indicator, self.studytype, area, self.name+'_'+area+'.json')
    
    @property
    def griddir(self):
        return os.path.join(self.config.cube_json, self.indicator, self.studytype, 'world', 'maps', self.name)

    def gridcsv(self, point):
        return os.path.join(self.griddir, str(point)+'.csv')


    def _init_axes_time(self, ds):
        self.climate_model_list = ds['climate_model'][:].tolist()
        self.impact_model_list = ds['impact_model'][:].tolist()
        self.climate_scenario_list = ds['ghg_concentration_scenario'][:].tolist()    
        self.years = ds['year'][:].tolist()    

        self.axes = {
          "timeslices_list": [(y-9,y+10) for y in self.years],
          "climate_scenario_list": self.climate_scenario_list,
          "climate_model_list": self.climate_model_list,
          "impact_model_list": self.impact_model_list,            
        }


    def _init_axes_temperature(self, ds):
        self.climate_model_list = ds['climate_model'][:].tolist()
        self.impact_model_list = ds['impact_model'][:].tolist()
        self.temperature_list = ds['temperature_change'][:].tolist()            

        self.axes = {
              "temperature_list": self.temperature_list,
              "climate_model_list": self.climate_model_list,
              "impact_model_list": self.impact_model_list,            
        }


    def _init_axes(self, area=None):
        with nc.Dataset(self.areanc(area or 'grid-level')) as ds:
            if 'temperature' in self.axis:
                self._init_axes_temperature(ds)
            else:
                self._init_axes_time(ds)


    def __repr__(self):
        return 'Variable("{}")'.format(self.name)



class Point:
    """One dot on the line plot = one map
    """
    def __init__(self, scenario, climate_model, impact_model, slice):
        self.climate_model = climate_model.replace('multi-model-median','median')
        self.impact_model = impact_model.replace('multi-model-median','median')
        self.slice = slice
        self.scenario = scenario
        
    def __str__(self):
        return '{scenario}_{climate_model}_{impact_model}_{slice}'.format(**vars(self))
  
    def __repr__(self):
        return 'Point({scenario}, {climate_model}, {impact_model}, {slice})'.format(**{k:repr(v) 
                                                                                       for k, v in vars(self).items()})


class Area:
    """country or other area
    """
    def __init__(self, code, name=None, mask=None, geom=None, properties=None):
        self.code = code
        self.name = name
        self.mask = mask
        self.geom = geom
        self.properties = properties

    def __str__(self):
        return '{} :: {}'.format(self.code, self.name)

    def __repr__(self):
        return 'Area({}, {})'.format(self.code, self.name)


def tolist(a):
    ' replace nan with none'
    return np.ma.array(a, mask=a.mask if hasattr(a, 'mask') else np.isnan(a)).tolist()


class JsonData:
    
    def __init__(self, variable, area, cube, cube_std=None):
        self.variable = variable
        self.area = area
        self.cube = cube
        self.cube_std = cube_std
        self.name = variable.ncvariable.replace('_',' ')+'s'
        self.metadata = {}

        assert len(variable.climate_model_list) == cube.climate_model.size, 'climate_model mismatch'
        assert len(variable.impact_model_list) == cube.impact_model.size, 'impact_model mismatch'
        #assert len(variable.climate_scenario_list) == cube.ghg_concentration_scenario.size, 'climate_scenario mismatch'

        self._vstemp = 'temperature' in self.variable.axis

        if self._vstemp:
            self.plot_type = "indicator_vs_temperature"
            self.plot_label_x = "Global warming level"
            self.plot_unit_x = "°C"    
            self.plot_title =  self.name + ' vs. Global warming level'

        else:

            self.plot_type = "indicator_vs_timeslices"    
            self.plot_label_x = "Time Slices"
            self.plot_unit_x = ""
            self.plot_title = self.name + ' vs. Time slices'
            self.metadata.update({"n_timeslices": len(self.variable.years)})


    def header(self):
        hdr = self.metadata.copy()
        hdr.update({
             'plot_type': self.plot_type,
              "indicator": self.variable.indicator,
              "variable": self.name,
              "assessment_category": self.variable.studytype,
              "area": self.area.code,
              "region": self.area.name,
              #"esgf_search_url": "https://esg.pik-potsdam.de/esg-search/search/?offset=0&limit=10&type=Dataset&replica=false&latest=true&project=ISIMIP2b&sector=Water+Global&distrib=false&replica=false&facets=world_region%2Cvariable%2Ctime_frequency%2Clicence%2Cproduct%2Cexperiment%2Cproject%2Ccountry%2Csector%2Cimpact_model%2Cperiod%2Cbias_correction%2Cdataset_type%2Cmodel%2Cvariable_long_name%2Cco2_forcing%2Csocial_forcing%2Cirrigation_forcing%2Cvegetation%2Ccrop%2Cpft%2Cac_harm_forcing%2Cdiaz_forcing%2Cfishing_forcing%2Cmf_region%2Cocean_acidification_forcing%2Cwr_station%2Cwr_basin%2Cmelting_forcing%2Cpopulation_forcing%2Cearth_model_forcing%2Cadaptation_forcing%2Cforestry_stand",
              "plot_title": self.plot_title,
              "plot_label_x": self.plot_label_x,
              "plot_unit_x": self.plot_unit_x,
              "plot_label_y": self.name,
              "plot_unit_y": "% of land area" if self.variable.exposure.startswith('land-area') else '% of population',
        })
        return hdr


    @staticmethod
    def _data_time(cube, variable, cube_std=None):       
        if cube_std is None:
            cube_std = da.zeros_like(cube)

        js = {}            
        for k, scenario in enumerate(variable.climate_scenario_list):
            js[scenario] = {}

            for i, gcm0 in enumerate(variable.climate_model_list):
                gcm = gcm0.replace('multi-model-median','overall')
                js[scenario][gcm] = {
                    'runs': {},
                }
                
                for j, impact in enumerate(variable.impact_model_list):
                    mean = cube.loc[(slice(None), scenario, gcm0, impact)].values
                    std = cube_std.loc[(slice(None), scenario, gcm0, impact)].values
                    js[scenario][gcm]['runs'][impact] = {
                        'mean': tolist(mean), 
                        'shading_upper_border': tolist(mean+std), 
                        'shading_lower_border': tolist(mean-std), 
                    }

                median = js[scenario][gcm]['runs'].pop('multi-model-median')
                median['median'] = median.pop('mean')                    
                js[scenario][gcm].update(median)
                        
        return js


    @staticmethod
    def _data_temperature(cube, variable, cube_std=None):       
        if cube_std is None:
            cube_std = da.zeros_like(cube)
        js = {}
        for i, gcm0 in enumerate(variable.climate_model_list):
            gcm = gcm0.replace('multi-model-median','overall')
            js[gcm] = {
                'runs': {},
            }
            for j, impact in enumerate(variable.impact_model_list):
                mean = cube.loc[(slice(None), gcm0, impact)].values
                std = cube_std.loc[(slice(None), gcm0, impact)].values                
                js[gcm]['runs'][impact] = {
                    'mean': tolist(mean),
                    'shading_upper_border': tolist(mean+std), 
                    'shading_lower_border': tolist(mean-std),                     
                }
            median = js[gcm]['runs'].pop('multi-model-median')
            median['median'] = median.pop('mean')
            js[gcm].update(median)

        return js

    def data(self):
        if self._vstemp:
            return self._data_temperature(self.cube, self.variable, self.cube_std)
        else:
            return self._data_time(self.cube, self.variable, self.cube_std)

    
    def todict(self):
        js = {}
        js.update(self.header())
        js.update(self.variable.axes)
        js['climate_model_list'] = [m for m in js['climate_model_list'] if m != 'multi-model-median']
        js['impact_model_list'] = [m for m in js['impact_model_list'] if m != 'multi-model-median']
        js.update({
            "data": self.data(),
        })
        return js    
  
_futures = ['rcp26', 'rcp60']

def _fill_historical(dima):
    m = dima.year <= 2005 
    for s in dima.ghg_concentration_scenario:
        if s in ['historical', 'piControl']:
            continue
        dima[m, s] = dima[m, 'historical']

def _mask_historical(dima):
    m = dima.year < 1990   # leave one historical point 
    for s in dima.ghg_concentration_scenario:
        if s in ['historical', 'piControl']:
            continue
        dima[m, s] = np.nan
    dima[dima.year > 2005, 'historical'] = np.nan


def _timeslice_average(dima, years):
    '''average time slices'''
    mean = da.stack([dima[y-9:y+10].mean(axis='year', skipna=True) for y in years], axis='year', keys=years)
    std = da.stack([dima[y-9:y+10].std(axis='year', skipna=True) for y in years], axis='year', keys=years)
    return mean, std


def _multimodelmedian(data):
    '''median of climate impact models'''
    climate_median = data.median(axis='climate_model', skipna=True)
    impact_median = data.median(axis='impact_model', skipna=True)
    overall = data.median(axis=('impact_model', 'climate_model'), skipna=True)

    data2 = da.DimArray(axes=[ax.values for ax in data.axes[:-2]]+[
        ['multi-model-median']+data.climate_model.tolist(), 
        ['multi-model-median']+data.impact_model.tolist()], dims=data.dims)
    data2.values[..., 1:, 1:] = data.values
    data2.values[..., 0, 0] = overall.values
    data2.values[..., 1:, 0] = impact_median.values
    data2.values[..., 0, 1:] = climate_median.values
    return data2


def _process_time_data(cube, years):
    _fill_historical(cube)
    mean, std = _timeslice_average(cube, years)
    _mask_historical(mean)
    _mask_historical(std)
    return _multimodelmedian(mean), _multimodelmedian(std)


def create_json(variable, area):
    '''laod create json file for an area'''
    cube = da.read_nc(variable.areanc(area.code), variable.ncvariable)

    if 'time' in variable.axis:
        cube, cube_std = _process_time_data(cube, variable.years)
    else:
        path = variable.areanc(area.code)
        pathvar = variable.ncvariable.replace('_','-')
        prefix = 'interannual-standard-deviation-of-'
        prefixvar = prefix.replace('-','_')
        cube_std = da.read_nc(path.replace(pathvar, prefix+pathvar), prefixvar+variable.ncvariable)

    return JsonData(variable, area, cube, cube_std)


def generate_variables(indicators, exposures, changes, axes, config=None):
    return [Variable(exposure+'-'+indicator+(('-'+change) if change else '')+'_ISIMIP-projections_'+axis, config=config)
                for indicator in indicators for exposure in exposures for change in changes for axis in axes]


def get_areas(geom=False, mask=False, mask_file=None, shape_file=None, code_name='ISO3'):
    import shapely.geometry as shg
    import fiona
    mask_file = mask_file or Config().mask_file
    shape_file = shape_file or Config().shape_file

    with nc.Dataset(mask_file) as ds:
        codes = sorted([v[2:] for v in ds.variables.keys() if v.startswith('m_')])

        countries = list(fiona.open(shape_file))
        areas = [Area(c['properties'][code_name], c['properties']['NAME'], geom=shg.shape(c['geometry']) if geom else None, properties=c['properties']) 
            for c in countries if c['properties'][code_name] in codes]

    if mask:
        with nc.Dataset(mask_file) as ds:
            for area in areas:
                area.mask = ds['m_'+area.code][:] == 1

    return sorted(areas, key=lambda a: a.code)



    #TODO: read data from json file

    # def load_data_from_ncline(self, area):
    #     data = {}
    #     for t, year in enumerate(self.years):
    #         weight = self.weighting.get_weight(t)
    #         aggregator = Aggregator(area.mask, weight)

    #         for k, scenario in enumerate(self.climate_scenario_list):
    #             for i, gcm in enumerate(self.climate_model_list):
    #                 for j, impact in enumerate(self.impact_model_list):
    #                     array = ds[self.variable.ncvariable][t, :, :, k, i, j]
    #                     data[(year, scenario, gcm, impact)] = aggregator(array)

    #     return data      
# class Aggregator:
#     def __init__(self, mask, weights):
#         self.mask = mask
#         self.weights = weights
#         self.total_per_cell = weights[mask]
#         self.total_per_area = np.sum(self.total_per_cell)

#     def __call__(self, array):
#         if hasattr(array, 'mask'):
#             array = array.filled(np.nan)  # TODO: check missing data issues more carefully        
#         res = np.sum(array[self.mask]*self.total_per_cell)/self.total_per_area
#         return res.tolist() if np.isfinite(res) else None



# class LandWeight:
#     def __init__(self, gridareafile=gridareafile):
#         with nc.Dataset(gridareafile) as ds:
#             self.cell_weights0 = ds['cell_area'][:].squeeze()

#     def get_weight(self, t=None):
#         return self.cell_weights0

#     def aggregate(self, variable):



# class PopolationWeight:
#     def __init__(self, years=[], totpopfile=totpopfile):
#         # self.cell_weight = nc.Dataset('griddata/pop_tot_2005soc_0.5deg.nc4')['pop_tot'][:].squeeze()
#         # years = []
#         self.cell_weight = nc.Dataset(totpopfile)['pop_tot']
#         self.years = years
#         self.cell_weight_indices = []

#         for y in years:
#             if y <= 1861:
#                 self.cell_weight_indices.append(0)
#             elif y >= 2005:
#                 self.cell_weight_indices.append(144)
#             else:
#                 self.cell_weight_indices.append(y-1861)         

#     def get_weight(selt, t=None):
#         if not self.cell_weight_indices:
#             return self.cell_weight[-1]

#         i = self.cell_weight_indices[t]
#         return self.cell_weight[i]


# def get_weighting(variable):
#     if variable.axes
#     if variable.exposure.startswith('land-area'):
#         return LandWeight()
#     else:
#         return PopolationWeight()



#     def get_weighting(self):
#         # weighting
#         if self.variable.exposure.startswith('land-area'):
#             weighting = LandWeight()
#         else:
#             weighting = PopolationWeight(self.years)
#         return weighting


    # def crunch_data_from_grid(self, ds, area):               
    #     data = {}
    #     for t, year in enumerate(self.years):
    #         weight = self.weighting.get_weight(t)
    #         aggregator = Aggregator(area.mask, weight)

    #         for k, scenario in enumerate(self.climate_scenario_list):
    #             for i, gcm in enumerate(self.climate_model_list):
    #                 for j, impact in enumerate(self.impact_model_list):
    #                     array = ds[self.variable.ncvariable][t, :, :, k, i, j]
    #                     data[(year, scenario, gcm, impact)] = aggregator(array)

    #     return data



# def main():
#     import argparse
#     parser = argparse.ArgumentParser()

#     areas = get_areas()

#     from config import exposures, indicators, changes, axes
#     parser.add_argument('--indicators', nargs='*', default=indicators, help='%(default)s')    
#     parser.add_argument('--exposure', nargs='*', default=exposures, help='%(default)s')    
#     parser.add_argument('--changes', nargs='*', default=changes, help='%(default)s')    
#     parser.add_argument('--axes', nargs='*', default=axes, help='%(default)s')    
#     parser.add_argument('--areas', nargs='*', default=[a.code for a in areas], help='%(default)s')    
#     o = parser.parse_args()

#     variables = generate_variables(o.indicators, o.exposures, o.changes, o.axes)

#     areas = [a for a in areas if a.code in o.areas]

#     with nc.Dataset(mask_file) as ds:
#         for area in areas:
#             area.mask = ds['m_'+area.code][:] == 1

#     for v in variables:
#         print(v)
#         gen = jsoncreator(v)
#         for area in areas:
#             js = gen.todict(area)



# if __name__ == '__main__':
#     main()