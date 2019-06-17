from spynnaker.pyNN.models.abstract_pynn_model import AbstractPyNNModel
from spinn_utilities.overrides import overrides
from spinnak_ear_application_vertex.spinnakear_application_vertex import \
    SpiNNakEarApplicationVertex
import numpy as np

# audio segment size
SEG_SIZE = 8

# biggest number of neurons for the ear model
MAX_NEURON_SIZE = 30000.0

# scale max
FULL_SCALE = 1.0

# number of ihcs per parent drnl
N_IHC = 5


class SpiNNakEar(AbstractPyNNModel):

    # default params
    default_population_parameters = {
        'audio_input': None,
        'fs': 22050.0,
        'n_channels': 3000,
        'pole_freqs': None,
        'param_file': None,
        'ear_index': 0,
        'scale': 1.0,
        'n_ihc': N_IHC
    }

    NAME = "SpikeSourceSpiNNakEar"
    MAGIC_1 = 2.0
    MAGIC_2 = 4.25
    MAGIC_3 = 30

    __slots__ = [
        #
        "_audio_input",
        #
        "_fs",
        #
        "_n_channels",
        #
        "_param_file",
        # left or right ear
        "_ear_index",
        # scale between all ear and mini versions
        "_scale",
        # human readable name of model
        "_model_name",
        # how many cores for ihc
        "_n_ihc"
    ]

    def __init__(
            self, audio_input=default_population_parameters['audio_input'],
            fs=default_population_parameters['fs'],
            n_channels=default_population_parameters['n_channels'],
            pole_freqs=default_population_parameters['pole_freqs'],
            param_file=default_population_parameters['param_file'],
            ear_index=default_population_parameters['ear_index'],
            scale=default_population_parameters['scale'],
            n_ihc=default_population_parameters['n_ihc']):

        if isinstance(audio_input, list):
            audio_input = np.asarray(audio_input)

        if len(audio_input.shape) > 1:
            raise Exception(
                "For binaural simulation please create separate "
                "SpiNNak-Ear populations (left/right)")

        if audio_input is None:
            audio_input = np.asarray([])

        self._audio_input = (
            audio_input[0:int(
                np.floor(len(audio_input) / SEG_SIZE) * SEG_SIZE)])
        self._audio_input = np.asarray(self._audio_input)

        if pole_freqs is None:
            max_power = min(
                [np.log10(self._model.fs / self.MAGIC_1), self.MAGIC_2])
            self._pole_freqs = np.flipud(np.logspace(
                np.log10(self.MAGIC_3), max_power, self._model.n_channels))
        else:
            self._pole_freqs = pole_freqs

        self._fs = fs
        self._n_channels = int(n_channels)
        self._pole_freqs = pole_freqs
        self._param_file = param_file
        self._ear_index = ear_index
        self._scale = scale
        self._model_name = self.NAME
        self._n_ihc = n_ihc

    @overrides(AbstractPyNNModel.create_vertex)
    def create_vertex(self, n_neurons, label, constraints):

        return SpiNNakEarApplicationVertex(
            n_neurons,  constraints, label, self)

    @staticmethod
    def spinnakear_size_calculator(scale=FULL_SCALE):
        return int(MAX_NEURON_SIZE * scale)

    @property
    def model_name(self):
        return self._model_name

    @property
    def audio_input(self):
        return self._audio_input

    @property
    def fs(self):
        return self._fs

    @property
    def n_channels(self):
        return self._n_channels

    @property
    def pole_freqs(self):
        return self._pole_freqs

    @property
    def param_file(self):
        return self._param_file

    @property
    def ear_index(self):
        return self._ear_index

    @property
    def scale(self):
        return self._scale

    @property
    def n_ihc(self):
        return self._n_ihc
