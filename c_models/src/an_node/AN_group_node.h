/*
 * IHC_AN.h
 *
 *  Created on: 1 Nov 2016
 *      Author: rjames
 */

#ifndef AN_group_node_H_
#define AN_group_node_H_

#define TIMER2_CONF        0x82
#define TIMER2_LOAD        0

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

typedef struct key_mask_table {
    uint32_t key;
    uint32_t mask;
    uint32_t offset;
} key_mask_table_entry;


#endif /*AN_group_node_H_ */
