
#################################################

# Problem 1. Compute Shortest Path Between any two coordinates in a city. You can solve this for your own city. 
# Assume that you can use network and load open street maps to convert it into a networkx graph. 
# If you don’t know how to do that, you can look it up. Once you load it up you need to code up dikstra search algorithm. 
# Network x has shortest path algorithm – do not use that.


######################################################


import numpy as np
import networkx as nx
import osmnx as ox
import heapq







def get_map_and_path(city:str,src_coords:tuple,target_coords:tuple,network_type="drive"):
    """
    Grab networkx graph from OSM then find and plot shortest path from source coordinates to target coordinates. 
    """



    G = ox.graph_from_place(city, network_type=network_type) #grab graph

    
    ox.plot_graph(G) #plot graph


   





if __name__ == "__main__":

    city = "Denver, Colorado, USA"

    ######################
    # POINTS OF INTEREST #
    ######################

    Colorado_State_Capitol = (39.7393, -104.9848)
    Denver_International_Airport = (39.856094, -104.673737)
    University_of_Denver = (39.6766,-104.9619)
    Coors_Field = (39.7559,-104.9942) #MLB baseball field 

    src_coords = University_of_Denver
    target_coords = Denver_International_Airport

    bbox = -122.43, 37.78, -122.41, 37.79

# create network from that bounding box
    G = ox.graph.graph_from_bbox(bbox, network_type="drive_service")


    ox.plot_graph(G) #plot graph


   






    
    ############################################################################
    # Some Checks to see if my dijkstra implementation is behaving as expected #
    ###########################################################################
    # nx_dist = nx.single_source_dijkstra_path_length(G,src,weight='length')

    # print(f"Distance check: {nx_dist[target] == dist}")

    # nx_path = nx.single_source_dijkstra_path(G,src,weight='length')

    # print(f"Path check: {nx_path[target]==S}")