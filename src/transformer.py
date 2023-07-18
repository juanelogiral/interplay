import transformer_methods
import inspect


class tr_metaclass(type):
    def __new__(cls, name, bases, dict):
        tr = super().__new__(cls, name, bases, dict)

        tr._available_transformers = {}
        tr._available_transformer_defaults = {}

        # Loads transformer models into a static variable of the Transformer class
        # _available_transformers has keys corresponding to function aliases and values containing function adresses
        # *_defaults contains the default values for the model, they are specified in the randmat variable default_args
        # a function is only loaded if its defaults are specified.

        for key, adress in inspect.getmembers(transformer_methods, inspect.isfunction):
            if key in transformer_methods.default_args:
                tr._available_transformers[key] = adress
                tr._available_transformer_defaults[
                    key
                ] = transformer_methods.default_args[key]
        return tr


class Transformer(metaclass=tr_metaclass):
    def __init__(self, name=None, **kwargs):
        self._model_alias = None
        self._model_function = None
        if name is not None:
            self.set_model(name, **kwargs)

    @classmethod
    def _check_arguments(cls, name, **kwargs):
        """
        Check that the keywords in `kwargs` are a subset of the transformer
        parameters
        """
        params_user = {}
        params_user.update(cls._available_transformer_defaults[name])
        params_user.update(kwargs)
        return set(params_user.keys()) == set(
            cls._available_transformer_defaults[name].keys()
        )

    def set_model(self, name, **kwargs):
        if not self._check_arguments(name, **kwargs):
            raise ValueError(
                "Arguments provided are not a subset of the model arguments."
            )

        # erase params of previous model
        if self._model_alias is not None:
            for key in self._available_transformer_defaults:
                delattr(self, key)

        self._model_alias = name
        self._model_function = self._available_transformers[name]

        # set default params
        for key, val in self._available_transformer_defaults.items():
            setattr(self, key, val)

        # override params
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __call__(self, network):
        self._model_function(network)
