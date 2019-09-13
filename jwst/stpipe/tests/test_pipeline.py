import os
from os.path import dirname, join, abspath
import sys

import numpy as np
from numpy.testing import assert_allclose
import pytest

from .. import Step, Pipeline, LinearPipeline

from .steps import WithDefaultsStep

from jwst.stpipe import crds_client
from jwst.extern.configobj.configobj import ConfigObj
from jwst import datamodels

from crds.core.exceptions import CrdsLookupError

# TODO: Test system call steps


def library_function():
    import logging
    log = logging.getLogger()
    log.info("This is a library function log")


class FlatField(Step):
    """
    An example flat-fielding Step.
    """

    # Load the spec from a file

    def process(self, science, flat):
        from ... import datamodels

        self.log.info("Removing flat field")
        self.log.info("Threshold: {0}".format(self.threshold))
        library_function()

        output = datamodels.ImageModel(data=science.data - flat.data)
        return output


class Combine(Step):
    """
    A Step that combines a list of images.
    """

    def process(self, images):
        from ... import datamodels

        combined = np.zeros((50, 50))
        for image in images:
            combined += image.data
        return datamodels.ImageModel(data=combined)


class Display(Step):
    """
    A Step to display an image.
    """

    def process(self, image):
        pass


class MultiplyBy2(Step):
    """
    A Step that does the incredibly complex thing of multiplying by 2.
    """

    def process(self, image):
        from ... import datamodels

        with datamodels.ImageModel(image) as dm:
            with datamodels.ImageModel() as dm2:
                dm2.data = dm.data * 2
                return dm2


class MyPipeline(Pipeline):
    """
    A test pipeline.
    """

    step_defs = {
        'flat_field': FlatField,
        'combine': Combine,
        'display': Display
        }

    spec = """
    science_filename = input_file()  # The input science filename
    flat_filename = input_file(default=None)     # The input flat filename
    output_filename = output_file()  # The output filename
    """

    def process(self, *args):
        from ... import datamodels

        science = datamodels.open(self.science_filename)
        if self.flat_filename is None:
            self.flat_filename = join(dirname(__file__), "data/flat.fits")
        flat = datamodels.open(self.flat_filename)
        calibrated = []
        calibrated.append(self.flat_field(science, flat))
        combined = self.combine(calibrated)
        self.display(combined)
        dm = datamodels.ImageModel(combined)
        dm.save(self.output_filename)
        return dm


class WithDefaultsPipeline(Pipeline):
    """A test pipeline that includes the WithDefaultsStep"""

    step_defs = {
        "step_1": WithDefaultsStep,
        "step_2": WithDefaultsStep
    }

    spec = """
        pipeline_par1 = string(default='default pipeline_par1 value')
        pipeline_par2 = string(default='default pipeline_par2 value')
        pipeline_par3 = string(default='default pipeline_par3 value')
        pipeline_par4 = string(default='default pipeline_par4 value')
    """

    def process(self, input):
        input = self.step_1(input)
        input = self.step_2(input)
        return input


def test_pipeline():
    pipeline_fn = join(dirname(__file__), 'steps', 'python_pipeline.cfg')
    pipe = Step.from_config_file(pipeline_fn)
    pipe.output_filename = "output.fits"

    assert pipe.flat_field.threshold == 42.0
    assert pipe.flat_field.multiplier == 2.0

    pipe.run()
    os.remove(pipe.output_filename)


def test_pipeline_python():
    steps = {
        'flat_field': {'threshold': 42.0}
        }

    pipe = MyPipeline(
        "MyPipeline",
        config_file=__file__,
        steps=steps,
        science_filename=abspath(join(dirname(__file__), 'data', 'science.fits')),
        flat_filename=abspath(join(dirname(__file__), 'data', 'flat.fits')),
        output_filename="output.fits")

    assert pipe.flat_field.threshold == 42.0
    assert pipe.flat_field.multiplier == 1.0

    pipe.run()
    os.remove(pipe.output_filename)


class MyLinearPipeline(LinearPipeline):
    pipeline_steps = [
        ('multiply', MultiplyBy2),
        ('multiply2', MultiplyBy2),
        ('multiply3', MultiplyBy2)
        ]


def test_partial_pipeline():
    pipe = MyLinearPipeline()

    pipe.end_step = 'multiply2'
    result = pipe.run(abspath(join(dirname(__file__), 'data', 'science.fits')))

    pipe.start_step = 'multiply3'
    pipe.end_step = None
    result = pipe.run(abspath(join(dirname(__file__), 'data', 'science.fits')))

    assert_allclose(np.sum(result.data), 9969.82514685, rtol=1e-4)
    os.remove('stpipe.MyLinearPipeline.fits')

def test_pipeline_commandline():
    args = [
        abspath(join(dirname(__file__), 'steps', 'python_pipeline.cfg')),
        '--steps.flat_field.threshold=47'
        ]

    pipe = Step.from_cmdline(args)

    assert pipe.flat_field.threshold == 47.0
    assert pipe.flat_field.multiplier == 2.0

    pipe.run()
    os.remove(pipe.output_filename)


def test_pipeline_commandline_class():
    args = [
        'jwst.stpipe.tests.test_pipeline.MyPipeline',
        '--logcfg={0}'.format(
            abspath(join(dirname(__file__), 'steps', 'log.cfg'))),
        # The file_name parameters are *required*
        '--science_filename={0}'.format(
            abspath(join(dirname(__file__), 'data', 'science.fits'))),
        '--output_filename={0}'.format(
            'output.fits'),
        '--steps.flat_field.threshold=47'
        ]

    pipe = Step.from_cmdline(args)

    assert pipe.flat_field.threshold == 47.0
    assert pipe.flat_field.multiplier == 1.0

    pipe.run()
    os.remove(pipe.output_filename)


def test_pipeline_commandline_invalid_args():
    from io import StringIO

    args = [
        'jwst.stpipe.tests.test_pipeline.MyPipeline',
        # The file_name parameters are *required*, and one of them
        # is missing, so we should get a message to that effect
        # followed by the commandline usage message.
        '--flat_filename={0}'.format(
            abspath(join(dirname(__file__), 'data', 'flat.fits'))),
        '--steps.flat_field.threshold=47'
        ]

    sys.stdout = buffer = StringIO()

    with pytest.raises(ValueError):
        Step.from_cmdline(args)

    help = buffer.getvalue()
    assert "Multiply by this number" in help


@pytest.mark.xfail(reason="Pipeline reference files aren't fully implemented")
@pytest.mark.parametrize(
    "command_line_pars, command_line_config_pars, pipeline_reference_pars, step_reference_pars, \
expected_pipeline_pars, expected_step_1_pars, expected_step_2_pars",
    [
        # If nothing else is present, we should use the spec defaults
        (
            None,
            None,
            None,
            None,
            {"pipeline_par1": "default pipeline_par1 value"},
            {"par1": "default par1 value"},
            {"par1": "default par1 value"}
        ),
        # Step reference file pars > spec defaults
        (
            None,
            None,
            None,
            {"par1": "step reference par1 value"},
            {"pipeline_par1": "default pipeline_par1 value"},
            {"par1": "step reference par1 value"},
            {"par1": "step reference par1 value"}
        ),
        # Pipeline reference file pars > step reference file pars
        (
            None,
            None,
            {
                "pipeline_par1": "pipeline reference pipeline_par1 value",
                "step_1": {"par1": "pipeline reference step_1 par1 value"},
                "step_2": {"par1": "pipeline reference step_2 par1 value"}
            },
            {"par1": "step reference par1 value"},
            {"pipeline_par1": "pipeline reference pipeline_par1 value"},
            {"par1": "pipeline reference step_1 par1 value"},
            {"par1": "pipeline reference step_2 par1 value"}
        ),
        # Command line config pars > pipeline reference pars
        (
            None,
            {
                "pipeline_par1": "config pipeline_par1 value",
                "step_1": {"par1": "config step_1 par1 value"},
                "step_2": {"par1": "config step_2 par1 value"}
            },
            {
                "pipeline_par1": "pipeline reference pipeline_par1 value",
                "step_1": {"par1": "pipeline reference step_1 par1 value"},
                "step_2": {"par1": "pipeline reference step_2 par1 value"}
            },
            {"par1": "step reference par1 value"},
            {"pipeline_par1": "config pipeline_par1 value"},
            {"par1": "config step_1 par1 value"},
            {"par1": "config step_2 par1 value"}
        ),
        # Command line override pars > all other pars
        (
            {
                "pipeline_par1": "override pipeline_par1 value",
                "step_1": {"par1": "override step_1 par1 value"},
                "step_2": {"par1": "override step_2 par1 value"}
            },
            {
                "pipeline_par1": "config pipeline_par1 value",
                "step_1": {"par1": "config step_1 par1 value"},
                "step_2": {"par1": "config step_2 par1 value"}
            },
            {
                "pipeline_par1": "pipeline reference pipeline_par1 value",
                "step_1": {"par1": "pipeline reference step_1 par1 value"},
                "step_2": {"par1": "pipeline reference step_2 par1 value"}
            },
            {"par1": "step reference par1 value"},
            {"pipeline_par1": "override pipeline_par1 value"},
            {"par1": "override step_1 par1 value"},
            {"par1": "override step_2 par1 value"},
        ),
        # Test complex merging of parameters
        (
            {
                "pipeline_par1": "override pipeline_par1 value",
                "step_1": {"par1": "override step_1 par1 value"},
                "step_2": {"par1": "override step_2 par1 value"}
            },
            {
                "pipeline_par2": "config pipeline_par2 value",
                "step_1": {"par2": "config step_1 par2 value"},
                "step_2": {"par2": "config step_2 par2 value"}
            },
            {
                "pipeline_par3": "pipeline reference pipeline_par3 value",
                "step_1": {"par3": "pipeline reference step_1 par3 value"},
                "step_2": {"par3": "pipeline reference step_2 par3 value"}
            },
            {"par4": "step reference par4 value"},
            {
                "pipeline_par1": "override pipeline_par1 value",
                "pipeline_par2": "config pipeline_par2 value",
                "pipeline_par3": "pipeline reference pipeline_par3 value",
                "pipeline_par4": "default pipeline_par4 value"
            },
            {
                "par1": "override step_1 par1 value",
                "par2": "config step_1 par2 value",
                "par3": "pipeline reference step_1 par3 value",
                "par4": "step reference par4 value",
                "par5": "default par5 value"
            },
            {
                "par1": "override step_2 par1 value",
                "par2": "config step_2 par2 value",
                "par3": "pipeline reference step_2 par3 value",
                "par4": "step reference par4 value",
                "par5": "default par5 value"
            }
        )
    ]
)
def test_pipeline_commandline_par_precedence(
    command_line_pars, command_line_config_pars, pipeline_reference_pars, step_reference_pars,
    expected_pipeline_pars, expected_step_1_pars, expected_step_2_pars, tmp_path, monkeypatch
):
    args = []

    pipeline_class_name = "jwst.stpipe.tests.test_pipeline.WithDefaultsPipeline"
    pipeline_config_name = "WithDefaultsPipeline"
    pipeline_reference_type = f"pars-{pipeline_config_name.lower()}"
    step_class_name = "jwst.stpipe.tests.steps.WithDefaultsStep"
    step_config_name = "WithDefaultsStep"
    step_reference_type = "pars-withdefaultsstep"
    input_path = join(dirname(__file__), "data", "science.fits")

    if command_line_config_pars:
        command_line_config_path = tmp_path/"with_defaults_pipeline.cfg"
        config = ConfigObj(str(command_line_config_path))
        config["class"] = pipeline_class_name
        config["name"] = pipeline_config_name
        config["steps"] = {}
        for key, value in command_line_config_pars.items():
            if isinstance(value, dict):
                config["steps"][key] = value
            else:
                config[key] = value
        config.write()
        args.append(str(command_line_config_path.absolute()))
    else:
        args.append(pipeline_class_name)

    args.append(input_path)

    if command_line_pars:
        for key, value in command_line_pars.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    args.append(f"--steps.{key}.{sub_key}={sub_value}")
            else:
                args.append(f"--{key}={value}")

    reference_file_map = {}
    for reference_pars, reference_type, class_name, config_name in [
        (
            pipeline_reference_pars,
            pipeline_reference_type,
            pipeline_class_name,
            pipeline_config_name
        ),
        (
            step_reference_pars,
            step_reference_type,
            step_class_name,
            step_config_name
        )
    ]:
        if reference_pars:
            reference_path = tmp_path/f"{reference_type}.asdf"
            parameters = {
                "class": class_name,
                "name": config_name,
            }
            for key, value in reference_pars.items():
                if isinstance(value, dict):
                    if "steps" not in parameters:
                        parameters["steps"] = {}
                    parameters["steps"][key] = value
                else:
                    parameters[key] = value
            model = datamodels.StepParsModel({"parameters": parameters})
            model.save(reference_path)

            reference_file_map[reference_type] = str(reference_path)

    def mock_get_reference_file(dataset, reference_file_type, observatory=None):
        if reference_file_type in reference_file_map:
            return reference_file_map[reference_file_type]
        else:

            raise CrdsLookupError(f"Error determining best reference for '{reference_file_type}'  = \
  Unknown reference type '{reference_file_type}'")
    monkeypatch.setattr(crds_client, "get_reference_file", mock_get_reference_file)

    pipeline = Step.from_cmdline(args)

    for key, value in expected_pipeline_pars.items():
        assert getattr(pipeline, key) == value

    for key, value in expected_step_1_pars.items():
        assert getattr(pipeline.step_1, key) == value

    for key, value in expected_step_2_pars.items():
        assert getattr(pipeline.step_2, key) == value
