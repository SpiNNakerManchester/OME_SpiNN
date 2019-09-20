from pacman.model.graphs.common import EdgeTrafficType
from pacman.model.graphs.machine import MachineEdge
from spynnaker.pyNN.models.abstract_models import AbstractFilterableEdge


class SpiNNakEarMachineEdge(MachineEdge, AbstractFilterableEdge):
    """ Machine Edge for SpiNNakear comms, to ensure not filterable
    """

    def __init__(
            self, pre_vertex, post_vertex,
            traffic_type=EdgeTrafficType.MULTICAST, label=None,
            traffic_weight=1):
        """
        :param pre_vertex: the vertex at the start of the edge
        :type pre_vertex:\
            :py:class:`pacman.model.graphs.machine.MachineVertex`
        :param post_vertex: the vertex at the end of the edge
        :type post_vertex:\
            :py:class:`pacman.model.graphs.machine.MachineVertex`
        :param traffic_type: The type of traffic that this edge will carry
        :type traffic_type:\
            :py:class:`pacman.model.graphs.common.EdgeTrafficType`
        :param label: The name of the edge
        :type label: str
        :param traffic_weight:\
            the optional weight of traffic expected to travel down this edge\
            relative to other edges (default is 1)
        :type traffic_weight: int
        """
        MachineEdge.__init__(
            self, pre_vertex, post_vertex, traffic_type, label, traffic_weight)
        AbstractFilterableEdge.__init__(self)

    def filter_edge(self, graph_mapper):
        return False
