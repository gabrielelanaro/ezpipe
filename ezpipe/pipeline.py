'''Crazy shit pipelining module this is pretty crazy'''
import dask

class Pipeline:

    def __init__(self):
        self.dsk = {}
        self.models = {}
        self.named_steps = {}

        self._checkpoints = []
        self._checkpoint_loaders = {}

    def add_step(self, start, end, func, name=None):
        if end in self.dsk:
            raise ValueError('"{}" is already in the computation graph'.format(end))
        if name is not None:
            func.__name__ = name

        if isinstance(start, str):
            self.dsk[end] = (func, start)
        elif isinstance(start, tuple):
            self.dsk[end] = (func, *start)
        else:
            raise ValueError('first argument must be either tuple or string')

    def add_map(self, start, end, func, name=None):
        if isinstance(start, tuple):
            raise ValueError('Can only map single-argument inputs')
        f = lambda arg: [func(a) for a in arg]
        f.__name__ = 'map({})'.format(func.__name__)
        return self.add_step(start, end, f, name)

    def add_transformer(self, start, end, transformer, name=None):
        ref = name if name is not None else (start, end)
        self.named_steps[ref] = (transformer, start, end)
        return self.add_step(start, end, transformer.transform)

    def add_model(self, start, end, model, name=None, mode='predict'):
        if mode == 'predict':
            func = model.predict
        elif mode == 'proba':
            func = model.predict_proba
        else:
            raise ValueError('mode should be predict or proba')

        ref = name if name is not None else (start, end)
        self.named_steps[ref] = (model, start, end)
        return self.add_step(start, end, func)

    def fit(self, refs, **kwargs):
        '''Fit one or more steps'''
        if not isinstance(refs, list):
            refs = [refs]

        for name in refs:
            model, start, end = self.named_steps[name]

            if end not in kwargs:
                kwargs[end] = None

            X = self.get(start, **kwargs)
            y = self.get(end, **kwargs)
            model.fit(X, y)

    def get(self, value, **kwargs):
        graph = self.dsk.copy()
        graph.update(kwargs)

        # We need to also compute checkpoints
        output = [value] + ['save_' + c for c in self._checkpoints]
        return dask.async.get_sync(graph, output)[0]

    def add_checkpoint(self, step, save=None, load=None):
        self.dsk['save_' + step] = (save, step)
        self._checkpoints.append(step)
        self._checkpoint_loaders[step] = load

    def get_checkpoint(self, step):
        return self._checkpoint_loaders[step]()

    def display(self):
        from dask.dot import dot_graph
        return dot_graph(self.dsk)
