
import tensorflow as tf

from supervisely_lib import logger, EventType
from tensorflow_core.python.training import training_util


class StepLogger(tf.train.SessionRunHook):
    """
    Hook that logs the current batch step to Supervise.ly
    """

    def __init__(self, total):
        """
        Total is the total number of batches to train for.
        """
        self.logger = logger
        self.total = total
        self._global_step_tensor = None

    def begin(self):
        self._global_step_tensor = training_util._get_or_create_global_step_read()
        self.logger.info('progress', extra={
            'event_type': EventType.PROGRESS,
            'subtask': 'Model training: ',
            'current': 0,
            'total': self.total
        })
        if self._global_step_tensor is None:
            raise RuntimeError("Global step should be created to use StopAtStepHook.")

    def after_run(self, run_context, run_values):
        step = run_context.session.run(self._global_step_tensor)
        self.logger.info('progress', extra={
            'event_type': EventType.PROGRESS,
            'subtask': 'Model training: ',
            'current': step,
            'total': self.total
        })
