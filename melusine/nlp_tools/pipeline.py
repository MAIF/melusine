from abc import abstractmethod

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from melusine.nlp_tools.base_melusine_class import BaseMelusineClass


class MelusinePipeline(Pipeline, BaseMelusineClass):
    FILENAME = "pipeline.json"
    STEPS_KEY = "steps"

    def __init__(self, steps, memory=None, verbose=None):
        super().__init__(steps=steps, memory=memory, verbose=verbose)
        self.memory = memory
        self.verbose = verbose

    def save(self, path, filename_prefix=None):
        steps = list()

        # Save pipeline steps
        n_steps = len(self.steps)
        for i, obj_params in enumerate(self.steps):
            name, obj = obj_params
            step_prefix = self.get_step_prefix(filename_prefix, i, n_steps)

            if hasattr(obj, "save"):
                obj.save(path, step_prefix)
            else:
                self.save_pkl_generic(obj, name, path, step_prefix)

            # Add object meta to Pipeline dict
            steps.append(self.get_obj_meta(obj, name))

        save_dict = self.__dict__.copy()
        save_dict[self.STEPS_KEY] = steps

        # Save Pipeline
        self.save_json(
            save_dict=save_dict,
            path=path,
            filename_prefix=filename_prefix,
        )

    @staticmethod
    def get_step_prefix(filename_prefix, i, n_steps):
        step_prefix = f"{i:0{len(str(n_steps))}d}"
        if filename_prefix:
            step_prefix += f"_{filename_prefix}"
        return step_prefix

    @classmethod
    def load(cls, path, filename_prefix: str = None):
        # Load the Pipeline config file
        config_dict = cls.load_json(path)
        steps = list()

        # Load steps meta data
        steps_meta = config_dict.pop(cls.STEPS_KEY)

        # Instantiate transformers
        n_steps = len(steps_meta)
        for i, obj_params in enumerate(steps_meta):
            step_prefix = cls.get_step_prefix(filename_prefix, i, n_steps)
            name, obj = cls.load_obj(obj_params, path=path, filename_prefix=step_prefix)
            steps.append((name, obj))

        return cls(steps=steps, **config_dict)


class MelusineTransformer(BaseMelusineClass, BaseEstimator, TransformerMixin):
    """
    Base transformer class.
    Melusine transformers have the following characteristics:
    - Can be added to a Melusine (or sklearn) pipeline
    - Can be saved and loaded
    """

    def fit(self, df, y=None):
        return self

    @abstractmethod
    def transform(self, df):
        raise NotImplementedError()
