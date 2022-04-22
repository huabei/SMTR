#!/usr/bin/bash
# touch hsa-3p.fold
# RNAfold -i $SMTR_DATA/miR_Sequence/mirgenedb/hsa-3p.fas -o hsa-3p.fold --noPS
# rm hsa-5p.fold
# mv *.fold $SMTR_DATA/miR_Sequence/mirgenedb/hairpin_sec/mature/3p
# touch hsa-5p.fold
# RNAfold -i $SMTR_DATA/miR_Sequence/mirgenedb/hsa-5p.fas -o hsa-5p.fold --noPS
# rm hsa-5p.fold
RNAfold -i $SMTR_DATA/miR_Sequence/mirgenedb/hsa-5p.fas --noPS > hsa-5p.fold
RNAfold -i $SMTR_DATA/miR_Sequence/mirgenedb/hsa-3p.fas --noPS > hsa-3p.fold
mv *.fold $SMTR_DATA/miR_Sequence/mirgenedb/hairpin_sec/mature

