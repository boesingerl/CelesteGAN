import json
import numpy as np
import jmespath
import cv2
import networkx as nx

from .level import Level, LevelRenderer, pad_square

class CelesteMap:
    """Class for reading full map information, to determine path from beginning of level to end (and render image showcasing it)
    """
    
    def __init__(self, level_path):
        
        with open(level_path, 'r') as handle:
            zone = json.load(handle)

        levels = jmespath.search("root.children[?name == 'levels'].children" , zone)[0]
        self.levels = [Level(x) for x in levels]

        self.level_dict = {lvl.name:lvl for lvl in self.levels}
        
        for curr_level in self.levels:
    
            for other_level in self.levels:

                if curr_level != other_level:

                    inter_up = CelesteMap.all_intersect(curr_level.openings_up, other_level.openings_down)
                    if len(inter_up) > 0 and curr_level.ymax == other_level.ymin:
                        curr_level.adj_up.append((other_level.name, inter_up[0]))

                    inter_right = CelesteMap.all_intersect(curr_level.openings_right, other_level.openings_left)
                    if len(inter_right) > 0 and curr_level.xmax == other_level.xmin:
                        curr_level.adj_right.append((other_level.name, inter_right[0]))

                    inter_down = CelesteMap.all_intersect(curr_level.openings_down, other_level.openings_up)
                    if len(inter_down) > 0 and curr_level.ymin == other_level.ymax:
                        curr_level.adj_down.append((other_level.name, inter_down[0]))

                    inter_left = CelesteMap.all_intersect(curr_level.openings_left, other_level.openings_right)
                    if len(inter_left) > 0 and curr_level.xmin == other_level.xmax:
                        curr_level.adj_left.append((other_level.name, inter_left[0]))
                        
        G = nx.DiGraph()

        G.add_nodes_from([l.name for l in self.levels])

        for l in self.levels:

            G.add_edges_from([(l.name,other,{'space':space}) for (other,space) in l.adj_down])
            G.add_edges_from([(l.name,other,{'space':space}) for (other,space) in l.adj_up])    
            G.add_edges_from([(l.name,other,{'space':space}) for (other,space) in l.adj_left])    
            G.add_edges_from([(l.name,other,{'space':space}) for (other,space) in l.adj_right])  
            
        self.G = G
        
        self.start_level = None
        self.end_level = None

        tmp_berry = None
        tmp_gem = None
        
        for lvl in self.levels:

            gbt = jmespath.search("children[?name == 'triggers'].children[] | [?name == 'goldenBerryCollectTrigger']", lvl.lvl)

            gb = jmespath.search("children[?name == 'entities'].children[] | [?name == 'goldenBerry']",  lvl.lvl) 

            bg = jmespath.search("children[?name == 'entities'].children[] | [?name == 'blackGem']",  lvl.lvl)

            if len(gbt) > 0:
                tmp_berry = lvl.name
        
            if len(bg) > 0:
                tmp_gem = lvl.name

            if len(gb) > 0:
                self.start_level = lvl.name
                
        if tmp_gem is not None:
            self.end_level = tmp_gem
            
        if tmp_berry is not None:
            self.end_level = tmp_berry

    def render_finish(self, current, nexxt):
    
        l0 = self.level_dict[current]
        l1 = self.level_dict[nexxt]

        for (lname, (mini, maxi)) in l0.adj_right:
            if lname == nexxt:
                actmin = l0.ymin + l0.height - mini
                actmax = l0.ymin + l0.height - maxi
                return (l0.renderer.render_finish(slice(actmax, actmin+1), -1))

        for (lname, (mini, maxi)) in l0.adj_left:
            if lname == nexxt:
                actmin = l0.ymin + l0.height - mini
                actmax = l0.ymin + l0.height - maxi
                return (l0.renderer.render_finish(slice(actmax, actmin+1), 0))

        for (lname, (mini, maxi)) in l0.adj_up:
            if lname == nexxt:

                actmin = mini - l0.xmin
                actmax = maxi - l0.xmin
                return (l0.renderer.render_finish(0, slice(actmin, actmax+1)))

        for (lname, (mini, maxi)) in l0.adj_down:
            if lname == nexxt:
                actmin = mini - l0.xmin
                actmax = maxi - l0.xmin
                return (l0.renderer.render_finish(-1, slice(actmin, actmax+1)))        
    
    def plot_graph(self):
        pos = nx.nx_agraph.graphviz_layout(self.G)
        nx.draw(self.G, pos=pos)
        nx.draw_networkx_labels(self.G,pos,font_size=7,font_family='sans-serif')

    @staticmethod
    def all_intersect(int1, int2):
    
        def get_intersection(interval1, interval2):
            new_min = max(interval1[0], interval2[0])
            new_max = min(interval1[1], interval2[1])
            return [new_min, new_max] if new_min <= new_max else None

        return [x for x in (get_intersection(y, z) for y in int1 for z in int2) if x is not None]