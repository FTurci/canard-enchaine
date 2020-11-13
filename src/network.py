from scipy.spatial import cKDTree
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt

class ContactNetwork(object):
    """Construct network of contacts from centres and contacts"""
    def __init__(self, centres,contacts, rcut=None, threshold_vol=None, percentile=90):
        self.centres = centres
        self.contacts = contacts
        self.contact_tree = cKDTree(contacts.coord)
        self.centre_tree = cKDTree(centres.stress_coord)


        if rcut==None:
            rcut = centres.radius.mean()*1.6

        neighs = self.contact_tree.query_ball_tree(self.centre_tree,rcut)
        closest = []
        closest_dists = []

        for i,n in enumerate(neighs):
            n = np.array(n)
        #     print(i,n
            if len(n)>0:
                dists = np.linalg.norm(centres.stress_coord[n]-contacts.coord[i], axis=1) 
        #         print(n,dists,dists.argsort())
                order = dists.argsort()[:2]
                closest.append(n[order])
                closest_dists.append(dists[order])
            else:
                closest.append([])
                closest_dists.append([])

        self.closest = closest
        self.closest_dists = closest_dists

        self.dists = np.concatenate(np.array([d for d in closest_dists if len(d)>0]))

        missing = [1 for n in closest if len(n)==0]

        print(" The number of contacts with no neighbouring particles is",sum(missing))

        self.graph = nx.Graph()
        if threshold_vol == None:
            self.threshold_vol = np.percentile(contacts.volume,percentile) #very important threshold: it decides how many edges form the network. Thousands of edges quickly become unmanegeable from the plotting point of view
        for i,e in enumerate(closest):
            if contacts.volume[i]>self.threshold_vol: 
                if len(e)>1:
                    self.graph.add_edge(e[0],e[1])#,weight=contacts.volume[i])#, weight=contacts.volume[i])
        self.graph.number_of_edges()

    def view_graph(self):
        pos = nx.spring_layout(self.graph)
        plt.figure(figsize=(16,16))
        nx.draw_networkx(self.graph, pos,node_size=10,with_labels=False,width=2)
        # plt.xlim(-0.25,0.25)
        # plt.ylim(-0.1,0.2)
        plt.savefig(f"figs/{G.number_of_edges()}edges.png", dpi=300)

    def export_components(self):
        Gcc = sorted(nx.connected_components(self.graph), key=len, reverse=True)
        N = len( np.concatenate([list(g) for g in Gcc]))
        with open(self.centres.stress_path+f'/components_threshold{self.threshold_vol}.xyz', 'w') as fw:
            fw.write(f'{N}\nParticles\n')
            for u,d in enumerate(list(Gcc)):
                ids = np.array(list(d))
                for i,ID in enumerate(ids):
                    p = self.centres.stress_coord[ID]
                    trace = self.centres.trace[ID]
                    fw.write(f'Type_{u} {p[0]} {p[1]} {p[2]} {trace}\n')
