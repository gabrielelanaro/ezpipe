from ezpipe import Pipeline
import pickle
import numpy as np

class Tokenizer:

    def transform(self, X):
        return [x.split() for x in X]

class LinearRegression:

    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return X

class CountVectorizer:

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        return X

def test_pipeline_simple():
    p = Pipeline()
    tokenizer = Tokenizer()
    p.add_transformer('input', 'output', tokenizer)
    result = p.get('output', input=['a b c', 'd e f'])
    assert result == ['a b c'.split(), 'd e f'.split()]

def test_pipeline_model():
    p = Pipeline()
    p.add_transformer('a', 'b', Tokenizer())
    p.add_model('b', 'c', LinearRegression(), name='regression')
    p.fit('regression', b=[[0], [1], [2]], c=[1, 2, 3])

def test_pipeline_dim_reduct():

    X = ['hello whale', 'whale narwhal']

    p = Pipeline()
    p.add_transformer('input', 'x', CountVectorizer(), name='vectorizer')
    p.add_model('x', 'y', LinearRegression(), name='regression')

    p.fit('vectorizer', input=X, x=None)
    p.fit('regression', input=X, y=[1, 2])
    print(p.get('y', input=['hello narwhal']))

def test_pipeline_map():
    X = ['A', 'Aa', 'B', 'Ba']
    p = Pipeline()
    p.add_map('input', 'lowercase', str.lower)


def test_fit_pipeline():
    # This is the definition of the pipeline
    p = Pipeline()
    p.add_step('input', 'normalized', func=lambda X: X)

    reg = LinearRegression()
    p.add_step('normalized', 'linearized', func=reg.predict)
    p.add_step('linearized', 'output', func=np.mean)

    # This is a fitting procedure
    X = p.get('normalized', input=np.random.rand(10, 3))
    y = np.random.rand(10)
    reg.fit(X, y)

    print(p.get('output', input=np.random.rand(10, 3)))

def test_step_multiarg():
    identity = lambda x: x
    sum2 = lambda a, b: a + b
    p = Pipeline()
    p.add_step('a', 'b', func=identity)
    p.add_step(('a', 'b'), 'c', func=sum2)

    assert p.get('c', a=1) == 2

def test_pipeline_complex():
    p = Pipeline()

    tokenizer = Tokenizer()
    p.add_step('input', 'tokenized', func=tokenizer)

    def chunk(X):
        return X
    p.add_step('tokenized', 'chunk', func=chunk)

    def tag(X):
        return X
    p.add_step('chunk', 'tag', func=tag)

    # We want to "leak" some data to disk
    p = Pipeline()
    p.add_step('input', 'tokenized', func=tokenizer)

    # Saving/loading intermediate step
    p.add_checkpoint('tokenized', save=lambda obj: pickle.dump(obj, open('/tmp/tokenized', 'wb')),
                                  load=lambda : pickle.load(open('/tmp/tokenized', 'rb')))

    p.add_step('tokenized', 'chunked', func=chunk)
    p.add_step('chunked', 'tagged', func=tag)

    # We can run the thing first
    p.get('tagged', input=['a b c', 'd e f']) # This will run from input

    # This will run from the leak
    p.get('tagged', tokenized=p.get_checkpoint('tokenized'))
