import os
import time
import shutil
import pathlib
import logging
import string
import tempfile
import random
import subprocess
from typing import List, Tuple, Union, Callable

import click
import jinja2 as j2
import numpy as np
import pytorch_lightning as pl


PATH = pathlib.Path(__file__).parent.absolute()
VERSION_PATH = os.path.join(PATH, 'VERSION')
EXPERIMENTS_PATH = os.path.join(PATH, 'experiments')
TEMPLATES_PATH = os.path.join(PATH, 'templates')

# Use this jinja2 environment to conveniently load the jinja templates which are defined as files within the
# "templates" folder of the package!
TEMPLATE_ENV = j2.Environment(
    loader=j2.FileSystemLoader(TEMPLATES_PATH),
    autoescape=j2.select_autoescape(),
)
TEMPLATE_ENV.globals.update(**{
    'zip': zip,
    'enumerate': enumerate
})

# This logger can be conveniently used as the default argument for any function which optionally accepts
# a logger. This logger will simply delete all the messages passed to it.
NULL_LOGGER = logging.Logger('NULL')
NULL_LOGGER.addHandler(logging.NullHandler())


# == CLI RELATED ==

def get_version():
    """
    Returns the version of the software, as dictated by the "VERSION" file of the package.
    """
    with open(VERSION_PATH) as file:
        content = file.read()
        return content.replace(' ', '').replace('\n', '')


# https://click.palletsprojects.com/en/8.1.x/api/#click.ParamType
class CsvString(click.ParamType):

    name = 'csv_string'

    def convert(self, value, param, ctx) -> List[str]:
        if isinstance(value, list):
            return value

        else:
            return value.split(',')


# == PYTORCH LIGHTNING ==

class BestModelRestorer(pl.Callback):
    """
    A callback class that keeps track of the best model according to a given metric and restores the state of the 
    model at which it achieved that best score at the end of the training.
    
    Note: the actual weights of the model at the end of the training will be replaced when this callback is used!
    """
    
    def __init__(self, monitor: str = "val_loss", mode: str = "min"):
        """
        :param monitor: The metric to monitor. This should be a key in the dictionary of metrics defined in the 
            `training_step` or `validation_step` methods of the LightningModule.
        :param mode: Whether to minimize or maximize the monitored metric. One of "min" or "max".
        """
        super().__init__()
        self.monitor = monitor
        if mode not in ["min", "max"]:
            raise ValueError("mode must be 'min' or 'max'.")
        self.mode = mode

        self.best_score = None
        self.best_state_dict = None
        self.best_time = None

    def on_fit_start(self, trainer, pl_module) -> None:
        """
        This method is called before starting to fit the model.
        
        Initialize the best score before starting the fit.
        """
        if self.mode == "min":
            self.best_score = float("inf")
        else:
            self.best_score = -float("inf")
        self.best_state_dict = None

    def on_validation_end(self, trainer, pl_module):
        """
        Called at the end of the validation loop. We check whether the monitored metric improved
        and if so, store the model state dict and log the improvement.
        """
        metrics = trainer.callback_metrics
        current_score = metrics.get(self.monitor)

        if current_score is None:
            # Metric not found, cannot update best score
            return

        if (
            (self.mode == "min" and current_score < self.best_score) or
            (self.mode == "max" and current_score > self.best_score)
        ):
            # Update best score and store model weights
            self.best_score = current_score
            self.best_state_dict = {
                k: v.cpu() for k, v in pl_module.state_dict().items()
            }
            self.best_time = time.time()

            # Log the new best score (if the logger is available)
            if trainer.logger is not None:
                trainer.logger.log_metrics({f"best_{self.monitor}": current_score}, step=trainer.global_step)
                
            # You could also print a message if desired:
            trainer.print(
                f"New best {self.monitor}={current_score:.4f} at step={trainer.global_step}."
            )

    def on_train_end(self, trainer, pl_module):
        """
        This method is called at the end of the model.
        
        At the end of training, restore the model to the best recorded state.
        """
        if self.best_state_dict is not None:
            pl_module.load_state_dict(self.best_state_dict)
            trainer.print(
                f"Restored the best model with {self.monitor}={self.best_score:.4f}."
            )


class GracefulTermination(pl.Callback):
    """
    A callback that stops the model training when receiving a keyboard interrupt (Ctrl+C) but allows the code execution to continue.
    """

    def on_keyboard_interrupt(self, trainer, pl_module):
        """
        This method is called when a keyboard interrupt is received.
        """
        trainer.should_stop = True
        trainer.print("Keyboard interrupt received. Stopping training gracefully.")



# == STRING UTILITY ==
# These are some helper functions for some common string related problems

def random_string(length: int,
                  chars: list = string.ascii_letters + string.digits
                  ) -> str:
    """
    Generates a random string with ``length`` characters, which may consist of any upper and lower case
    latin characters and any digit.

    The random string will not contain any special characters and no whitespaces etc.

    :param length: How many characters the random string should have
    :param chars: A list of all characters which may be part of the random string
    :return:
    """
    return ''.join(random.choices(chars, k=length))


# == LATEX UTILITY ==
# These functions are meant to provide a starting point for custom latex rendering. That is rendering latex
# from python strings, which were (most likely) dynamically generated based on some kind of experiment data

def render_latex(kwargs: dict,
                 output_path: str,
                 template_name: str = 'article.tex.j2'
                 ) -> None:
    """
    Renders a latex template into a PDF file. The latex template to be rendered must be a valid jinja2
    template file within the "templates" folder of the package and is identified by the string file name
    `template_name`. The argument `kwargs` is a dictionary which will be passed to that template during the
    rendering process. The designated output path of the PDF is to be given as the string absolute path
    `output_path`.

    **Example**

    The default template for this function is "article.tex.j2" which defines all the necessary boilerplate
    for an article class document. It accepts only the "content" kwargs element which is a string that is
    used as the body of the latex document.

    .. code-block:: python

        import os
        output_path = os.path.join(os.getcwd(), "out.pdf")
        kwargs = {"content": "$\text{I am a math string! } \pi = 3.141$"
        render_latex(kwargs, output_path)

    :raises ChildProcessError: if there was ANY problem with the "pdflatex" command which is used in the
        background to actually render the latex

    :param kwargs:
    :param output_path:
    :param template_name:
    :return:
    """
    with tempfile.TemporaryDirectory() as temp_path:
        # First of all we need to create the latex file on which we can then later invoke "pdflatex"
        template = TEMPLATE_ENV.get_template(template_name)
        latex_string = template.render(**kwargs)
        latex_file_path = os.path.join(temp_path, 'main.tex')
        with open(latex_file_path, mode='w') as file:
            file.write(latex_string)

        # Now we invoke the system "pdflatex" command
        command = (f'pdflatex  '
                   f'-interaction=nonstopmode '
                   f'-output-format=pdf '
                   f'-output-directory={temp_path} '
                   f'{latex_file_path} ')
        proc = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise ChildProcessError(f'pdflatex command failed! Maybe pdflatex is not properly installed on '
                                    f'the system? Error: {proc.stdout.decode()}')

        # Now finally we copy the pdf file - currently in the temp folder - to the final destination
        pdf_file_path = os.path.join(temp_path, 'main.pdf')
        shutil.copy(pdf_file_path, output_path)


def latex_table_element_mean(values: List[float],
                             template_name: str = 'table_element_mean.tex.j2',
                             vertical: bool = True,
                             raw: bool = False,
                             ) -> str:
    if raw:
        mean, std = values
    else:
        mean = np.mean(values)
        std = np.std(values)

    template = TEMPLATE_ENV.get_template(template_name)
    return template.render(
        mean=mean,
        std=std,
        vertical=vertical
    )


def latex_table_element_median(values: List[float],
                               upper_quantile: float = 0.75,
                               lower_quantile: float = 0.25,
                               include_variance: bool = True,
                               template_name: str = 'table_element_median.tex.j2') -> str:
    median = np.median(values)
    upper = np.quantile(values, upper_quantile)
    lower = np.quantile(values, lower_quantile)

    template = TEMPLATE_ENV.get_template(template_name)
    return template.render(
        median=median,
        upper=upper,
        lower=lower,
        include_variance=include_variance
    )


def latex_table(column_names: List[str],
                rows: List[Union[List[float], str]],
                content_template_name: str = 'table_content.tex.j2',
                table_template_name: str = 'table.tex.j2',
                list_element_cb: Callable[[List[float]], str] = latex_table_element_mean,
                prefix_lines: List[str] = [],
                caption: str = '',
                ) -> Tuple[str, str]:

    # ~ Pre Processing the row elements into strings
    string_rows = []
    for row_index, row in enumerate(rows):
        string_row = []
        for element in row:
            if isinstance(element, str):
                string_row.append(element)
            if isinstance(element, list) or isinstance(element, np.ndarray):
                string = list_element_cb(element)
                string_row.append(string)

        string_rows.append(string_row)

    alignment = ''.join(['c' for _ in column_names])

    # ~ Rendering the latex template(s)

    content_template = TEMPLATE_ENV.get_template(content_template_name)
    content = content_template.render(rows=string_rows)

    table_template = TEMPLATE_ENV.get_template(table_template_name)
    table = table_template.render(
        alignment=alignment,
        column_names=column_names,
        content=content,
        header='\n'.join(prefix_lines),
        caption=caption,
    )

    return content, table