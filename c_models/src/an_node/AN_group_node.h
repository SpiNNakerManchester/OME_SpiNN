/*
 * Copyright (c) 2019-2020 The University of Manchester
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef AN_group_node_H_
#define AN_group_node_H_

//! \ brief marker to fill out api for packet sending without payload
#define PARAM_FILLER 0

//! \brief table entry
typedef struct key_mask_table_entry {
    uint32_t key;
    uint32_t mask;
    uint32_t offset;
} key_mask_table_entry;

//! \brief params from parameter region
typedef struct params_struct {
    int n_children;
    int has_key;
    int key;
    int is_final_row;
    int n_atoms;
} params_struct;

//! \brief key map struct
typedef struct key_map_struct {
    key_mask_table_entry *entries;
} key_map_struct;

//! \brief priorities
typedef enum priorities {
    MC = -1, TIMER = 0, USER = 1, SDP_PRIORITY = 1,
    DMA_TRANSFER_DONE_PRIORITY = 0
} priorities;

//! \brief provenance data
typedef enum provenance_items {
    N_SPIKES_TRANSMITTED
} provenance_items;

//! \brief data spec regions
typedef enum regions {
    SYSTEM,
    PARAMS,
    KEY_MAP,
    PROVENANCE
} regions;

#endif /*AN_group_node_H_ */