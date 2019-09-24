# Copyright (c) 2019-2020 The University of Manchester
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
