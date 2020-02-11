import json

def _hashsum(string, l=6):
	return hashlib.sha1(string.encode()).hexdigest()[:l] 


def figcode(data, **kwargs):
	"code based on file name and figure arguments"
	kwargs['filename'] = data['filename']
	string = json.dumps(kwargs, sort_keys=True)
	return _hashsum(string)
	# return os.path.splitext(data['filename'])[0].lower()


def _lineplot(data, x=None, scenario=None, climate=None, impact=None, shading=False, color=None):
	import matplotlib.pyplot as plt
	lines = data.filter_lines(scenario, climate, impact)

	if data.plot_type == 'indicator_vs_timeslices':
		x = np.arange(len(data.x))  # string !
	else:
		x = data.x

	fig, (ax, ) = plt.subplots(1,1)
	ix = data.index(x)
	for l in lines:
		if shading:
			ax.fill_between(x, l['lower'], l['upper'], color=color)
		ax.plot(x, l['y'], color=color)

	ax.set_xticks(x)
	ax.set_xtick_labels(data.x)
	return fig, ax


class LinePlot:
	backends = ['mpl', 'mpld3']
	make = _lineplot


	def __init__(self, backend, makefig=True):
		self.backend = backend
		self.makefig = makefig


	def __call__(self, data, **kwargs):
		if self.makefig:
			self.make_figs(data, kwargs, [self.backend])
		return self.insert_cmd(data, kwargs)


	def figpath(self, data, kwargs, backend=None):
		ext = {'mpl': 'png', 'mpld3': 'json'}[backend or self.backend]
		return figcode(data, **kwargs) + '.' + ext


	def insert_cmd(self, data, kwargs, backend=None):
		backend = backend or self.backend
		if backend == 'mpl':
			return '![{}]({})'.format(self.caption, self.figpath(data, kwargs, backend))
		elif backend == 'mpld3':
			return '<div id={}></div>'.format(self.figcode(data, kwargs))


	def make_figs(self, data, kwargs, backends=None):
		"custom make figs that reuse same figure"
		backends = backends or self.backends
		if not backends : return
		figcode = figcode(data, kwargs)

		fig, axes = _lineplot(data, **kwargs)
		for backend in backends:		
			if backends == 'mpl':
				fig.savefig(self.figpath('mpl'), dpi=100)
			elif backends == 'mpld3':
				js = mpld3.fig_to_json(fig)
				json.dump(js, open(self.figpath('mpld3'), 'w'))
		fig.close()