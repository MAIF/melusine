import os
from sklearn.pipeline import Pipeline

from melusine.core.saver_mixin import SaverMixin


class MelusinePipeline(Pipeline, SaverMixin):
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
            step_prefix = self.get_step_prefix(name, filename_prefix, i, n_steps)

            if isinstance(obj, MelusinePipeline):
                step_path = os.path.join(path, step_prefix)
                obj.save(step_path)
            elif hasattr(obj, "save"):
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
    def get_step_prefix(name, filename_prefix, i, n_steps):
        prefix = ""
        if filename_prefix:
            prefix += f"{filename_prefix}_"

        prefix += f"{i:0{len(str(n_steps))}d}_{name}"
        return prefix

    @classmethod
    def load(cls, path, filename_prefix: str = None):
        # Load the Pipeline config file
        config_dict = cls.load_json(path, filename_prefix=filename_prefix)
        steps = list()

        # Load steps meta data
        steps_meta = config_dict.pop(cls.STEPS_KEY)

        # Instantiate transformers
        n_steps = len(steps_meta)
        for i, obj_params in enumerate(steps_meta):
            name = obj_params[cls.SAVE_NAME]
            class_name = obj_params[cls.SAVE_CLASS]
            step_prefix = cls.get_step_prefix(name, filename_prefix, i, n_steps)

            # Pipeline composition (Load a MelusinePipeline object)
            if class_name == cls.__name__:
                step_path = os.path.join(path, step_prefix)
                obj = cls.load(step_path)

            # Load regular transformer (Melusine or sklearn)
            else:
                name, obj = cls.load_obj(
                    obj_params, path=path, filename_prefix=step_prefix
                )
            steps.append((name, obj))

        return cls(steps=steps, **config_dict)
