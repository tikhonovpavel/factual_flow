class ContextWrapper:
    def __init__(self, klass, **kwargs):
        self.klass = klass
        self.default_kwargs = kwargs

    def __call__(self, **kwargs):
        combined_kwargs = {**self.default_kwargs, **kwargs}
        return self.klass(**combined_kwargs)
