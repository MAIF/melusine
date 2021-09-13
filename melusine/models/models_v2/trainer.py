import logging
import pandas as pd
from tensorflow.keras.optimizers import Adam
from melusine.models.models_v2.base_model import BaseMelusineModel


logger = logging.getLogger(__name__)


class MelusineTrainer:
    """
    MelusineTrainer class to train Keras deep learning models.
    This class contains all the setup necessary to train a model (Optimizers, Losses, etc)
    but useless to use the model for inference.
    """

    def __init__(
        self,
        melusine_model: BaseMelusineModel,
        epochs: int = 5,
        batch_size: int = 256,
        loss: str = "categorical_crossentropy",
        learning_rate: float = 5e-5,
    ):
        """

        Parameters
        ----------
        melusine_model: BaseMelusineModel
            Melusine model to be trained
        epochs: int
            Number of epochs for training
        batch_size: int
            Number of elements in a training batch
        loss: str
            Loss function for training
        learning_rate: float
            Learning rate
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.melusine_model = melusine_model
        self.loss = loss
        self.learning_rate = learning_rate

    def train(self, x: pd.DataFrame, y: pd.Series = None, validation_data=None) -> None:
        """
        Train the Deep Learning model

        Parameters
        ----------
        x: pd.DataFrame
            Input features
        y: pd.Series
            Input labels
        validation_data: Tuple()  # TODO: docstring
        """

        logger.info("Fit Melusine model parameters")
        self.melusine_model.fit(x, y)

        logger.info("Encode train labels")
        y_cat = self.melusine_model.encode_labels(y)

        logger.info("Transform train features")
        x_input = self.melusine_model.transform(x)

        logger.info("Train model")
        optimizer = Adam(learning_rate=self.learning_rate)
        self.melusine_model.model.compile(
            optimizer=optimizer, loss=self.loss, metrics=["accuracy"]
        )
        self.melusine_model.model.fit(
            x_input,
            y_cat,
            batch_size=self.batch_size,
            epochs=self.epochs,
            # validation_data=validation_data,
        )
        logger.info("Finished model training")
