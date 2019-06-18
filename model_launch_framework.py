import spinnaker_graph_front_end as g

from spinnak_ear.spinnak_ear_machine_vertices.ome_machine_vertex import \
    OMEMachineVertex
from spinnak_ear.spinnak_ear_machine_vertices.drnl_machine_vertex import \
    DRNLMachineVertex
from spinnak_ear.spinnak_ear_machine_vertices.ihcan_machine_vertex import \
    IHCANMachineVertex
import model_binaries

from pacman.model.constraints.placer_constraints\
    .chip_and_core_constraint import ChipAndCoreConstraint
from pacman.model.graphs.machine import MachineEdge

from spinnman.model.enums.cpu_state import CPUState

from spinn_front_end_common.utilities import globals_variables

import numpy
import logging
import time
import math

logger = logging.getLogger(__name__)

def calculate_additional_chips(num_rows,num_macks):
    acc=0.
    for i in range(num_rows-1):
        acc += num_macks**i
    return int(numpy.ceil(acc/16.))

def run_model(
        data, n_chips=None,n_drnl=0,pole_freqs=[4000],n_ihcan=0,
        fs=44100,resample_factor=1,num_macks=4,seg_size=96,bitfield=True,
        rt=True,profile=True):

    # calculate number of mack tree rows above OME (final row is all DRNL instances)
    n_mack_tree_rows = int(numpy.ceil(math.log(len(pole_freqs),num_macks)))
    #calculate remainder drnls, if non-zero once this many are generated all following mack-->drnl connections will be 1 to 1
    remainder_macks =(num_macks**(n_mack_tree_rows)) - len(pole_freqs)

    additional_chips = calculate_additional_chips(num_macks=num_macks, num_rows=n_mack_tree_rows)

    # Set up the simulation
    g.setup(n_chips_required=n_chips+additional_chips, model_binary_module=model_binaries)
    # Get the number of cores available for use
    machine = g.machine()
    boards = dict()
    #changed to lists to ensure data is read back in the same order that vertices are instantiated
    omes=list()
    drnls=list()
    ihcans=list()
    macks=list()

    delays =  numpy.round(len(pole_freqs)*numpy.random.rand(len(pole_freqs)))
    if n_ihcan>0:
        #create unique seeds for IHCAN instances
        n_ihcans = n_chips*n_drnl*n_ihcan
        random_range = numpy.arange(n_ihcans*4,dtype=numpy.uint32)
        seeds = numpy.random.choice(random_range,int(n_ihcans*4),replace=False)
    seed_index = 0
    cf_index = 0
    count = 0

    #OME is on ethernet chip for live streaming
    for chip in machine.ethernet_connected_chips:
        # create OME
        ome = OMEMachineVertex(data, fs, len(pole_freqs), rt=rt, profile=profile)
        g.add_machine_vertex_instance(ome)
        # constrain placement to local chip
        ome.add_constraint(ChipAndCoreConstraint(chip.x, chip.y))
        omes.append(ome)
        boards[chip.x, chip.y] = chip.ip_address
        ome_chip_xy = [(chip.x, chip.y)]
        break

    for chip in machine.chips:
        if count >= n_chips:
            break
        #else:
        elif (chip.x, chip.y) not in ome_chip_xy:
            #create DRNLs
            for i in range(n_drnl):

                CF=pole_freqs[cf_index]
                drnl=DRNLMachineVertex(ome, CF, delays[cf_index], profile=profile)
                g.add_machine_vertex_instance(drnl)
                # constrain placement to local chip
                drnl.add_constraint(ChipAndCoreConstraint(chip.x, chip.y))
                drnls.append(drnl)
                cf_index=cf_index+1

                for j in range(n_ihcan):
                    ihcan = IHCANVertex(drnl,resample_factor,
                                        seeds[seed_index:seed_index+4],
                                        bitfield=bitfield,profile=profile)
                    seed_index += 4
                    g.add_machine_vertex_instance(ihcan)
                    # constrain placement to local chip
                    ihcan.add_constraint(ChipAndCoreConstraint(chip.x, chip.y))
                    ihcans.append(ihcan)
                    # Add an edge from the DRNL to the IHCAN, to send the data
                    g.add_machine_edge_instance(
                        MachineEdge(drnl, ihcan),
                        drnl.data_partition_name)

                    # Add an edge from the IHCAN to the DRNL,
                    # to send acknowledgement
                    g.add_machine_edge_instance(
                        MachineEdge(ihcan, drnl),
                        drnl.acknowledge_partition_name)

                # Add an edge from the OME to the DRNL, to send the data
                g.add_machine_edge_instance(
                    MachineEdge(ome, drnl),
                    ome.data_partition_name)

            count=count+1

    # here we define the MC ack tree
    parents = []
    parents.append(ome)
    drnls_index=0
    target_drnls = len(drnls)

    for k in range(n_mack_tree_rows):
        # generate row mack vertices
        if k<n_mack_tree_rows-1:
            # clear macks list
            macks[:] = []
        parent_count=0
        first=1
        for parent in parents:
            #reduced mack connections to account for remainder
            if k == (n_mack_tree_rows-1) and remainder_macks>0 and len(drnls)>num_macks:
                #calculate the distribution of the drnls among the
                # available macks (min of 1, max of 4 drnls per mack)
                #find available macks
                avail_macks=len(parents)-parent_count

                if avail_macks <= target_drnls and first:
                    n_macks = remainder_macks % num_macks
                    first=0
                elif avail_macks * n_macks > target_drnls and avail_macks>1:
                    n_macks = int((avail_macks*n_macks)/target_drnls)
                else:
                    n_macks = int(target_drnls/avail_macks)
                #clip n_macks
                if n_macks > num_macks:
                    n_macks = num_macks
                elif n_macks < 1:
                    n_macks = 1
                #update count and drnls remaining
                parent_count += 1

            #MC Ack for very small simulations
            elif len(drnls)<num_macks:
                n_macks=len(drnls)
            else:
                n_macks=num_macks
            for l in range(n_macks):
                #final row drnl connections
                if k>= n_mack_tree_rows -1:
                    mack=drnls[drnls_index]
                    drnls_index += 1
                    target_drnls -=1
                    #register this drnl instance to parent node list
                    mack.register_ack_processor(parent)
                else:
                    mack = MCackMachineVertex(parent)
                    g.add_machine_vertex_instance(mack)
                    macks.append(mack)

                # Add an edge to send r2s data
                g.add_machine_edge_instance(
                    MachineEdge(parent, mack),
                    parent.command_partition_name)
                # Add an edge to send ack
                g.add_machine_edge_instance(
                    MachineEdge(mack, parent),
                    parent.acknowledge_partition_name)

        # reupdate parents list with this row's macks
        parents[:]=macks

    if target_drnls!=0:
        raise ValueError('failed to setup MC ack tree structure')

# Run the simulation
    if rt:
        latency = 3 * (96./fs)
        duration = (len(data)/fs) * 1000.
    else:
        duration = 2000.
    #g.run(duration)
    g.transceiver().set_watch_dog(False)

    g.run_until_complete()

    # Get the data back
    samples = list()
    for drnl in drnls:
        samples.append(drnl.read_samples(g.buffer_manager()))
    samples = numpy.hstack(samples)

    # Close the machine
    g.stop()
    profiles = []
    if profile:
        drnl_profiles=[]
        ihc_profiles=[]
        profiles.append(ome._process_profile_times)
        for drnl in ome._drnl_vertices:
            drnl_profiles.append(drnl._process_profile_times)
            for ihc in drnl._ihcan_vertices:
                ihc_profiles.append(ihc._process_profile_times)
        profiles.append(numpy.asarray(drnl_profiles))
        profiles.append(numpy.asarray(ihc_profiles))
    print "channels running: ",len(drnls)
    print "output data: {} fibres with length {}".format(len(ihcans)*2,len(samples))
    if(len(samples) != len(ihcans)*2*numpy.floor(len(data)/seg_size)*seg_size):
        print "samples length {} isn't expected size {}".format(len(samples),len(ihcans)*2*numpy.floor(len(data)/seg_size)*seg_size)

    return samples,numpy.asarray(profiles)
