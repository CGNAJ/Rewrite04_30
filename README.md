# Rewrite04_30

File format：
In output csv file, each row is an event. 
The columns from 1-42 are strip number for 6 RPC layers. 7 bits of data for one RPC layer.
The column 43 is the average strip number super-cluster in this candidate event on RPC1, column 44 for RPC2's, column 45 for RPC3's.
In one RPC layer(7 bits of data):
        The first column is the strip number of main muon hit.(-1 means this signal has not been received because of strip-efficiency)
        The second column is for the adjacent muon hit from side-effect.(-1 means this signal has not been received because of strip-efficiency)
        Column 3-7 are bits for noise hit, noise hit with the same strip number as muon hit will be rejected, if the number of noise hits is less than 5, using 0 to fill the bits.

Print function:
Strip_list: Row 137 in code. The primitive event containing one muon, 42 bits of the strip number on all layers
Cluster_strip: Row 372. Single-layer RPC cluster for 6 layers. For example:
                        [[[138, 1, 0]], [[138, 1, 1]], [[220, 1, 2], [152, 1, 2]], [[151.5, 2, 3]], [[324, 1, 4], [200, 1, 4], [74, 1, 4]], [[199, 1, 5], [190, 1, 5]]]
                         <<  RPC1-1 >>  <<  RPC1-2 >>  <<         RPC2-1       >>  <<   RPC2-2  >>  <<              RPC3-1              >>  <<       RPC3-2         >>
                           1 cluster       1 cluster             2 cluster             1 cluster                  3 cluster                         2 cluster  
               For RPC2-2:   [151.5, 2, 3]
                              That means the strip number of this single-layer cluster is 151.5.
                              There are 2 hits in this single-layer cluster.
                              This single-layer cluster is located on the 3th layer(RPC2-1)
                              
Cluster_number: Row 372. Show the number of single-layer cluster. 
                   Example:     [1, 1, 2, 1, 3, 2]  2 single-layer clusters on RPC2-1
                                                    3 single-layer clusters on RPC3-1
                                                    2 single-layer clusters on RPC3-2
                                                    
RPC_cluster: Row 379-381.Show the super cluster on one doublet layer.
                  Example: 1 [[156.75, 2, 3, 2], [129.0, 1, 1, 1]]
                          They are super clusters on RPC1.
                          2 super clusters.
                          For the first super cluster [156.75, 2, 3, 2]: The average strip number is 156.75
                                                                         There are 2 single-layer clusters in this super cluster.
                                                                         There are 3 hits in this super cluster.
                                                                         Those single-layer clusters are from 2 different RPC layers.
                          For the second super cluster [129.0, 1, 1, 1]: The average strip number is 129.
                                                                         There are 1 single-layer clusters in this super cluster.
                                                                         There are 1 hits in this super cluster.
                                                                         Those single-layer cluster is from 1 RPC layer.
