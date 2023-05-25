"""Utils to read level from json and convert to image"""

import json
import numpy as np
import jmespath
import matplotlib.pyplot as plt
from glob import glob
import cv2
import numpy.lib.recfunctions as nlr
import tempfile
import os
import subprocess

class LevelEncoder:
    orig_path = os.path.dirname(os.path.realpath(__file__))
    
    def write_level(path, dic, exec_path=None):
        fp = tempfile.NamedTemporaryFile(mode='w', delete=False)
        json.dump(dic, fp)

        fp.flush()
        os.fsync(fp)

        exec_path = exec_path if exec_path is not None else  os.path.join(LevelEncoder.orig_path, 'json2map.rb')
        subprocess.Popen(['ruby', exec_path, fp.name , path]).wait()

        fp.close()
        os.unlink(fp.name)
    
    def read_level(path, exec_path=None):
        fp = tempfile.NamedTemporaryFile(mode='r', delete=False)
    
        exec_path = exec_path if exec_path is not None else  os.path.join(LevelEncoder.orig_path, 'map2json.rb')
        subprocess.Popen(['ruby', exec_path, path, fp.name]).wait()

        fp.flush()
        os.fsync(fp)

        level_map = json.load(fp)
        
        fp.close()
        os.unlink(fp.name)

        return level_map

def pad_square(img, size, maxdim=1, mode='constant', add_values={}):
    dims = img.shape[:maxdim+1]
    
    pads = [((size-dim)//2,size-dim-(size-dim)//2) for dim in dims] + [(0,0) for _ in img.shape[maxdim+1:]]
    return np.pad(img, pads, mode=mode, **add_values)

def decode_rle(l, nums, width, ignore_list=[13]):
    
    final = []
    tmp = []
    
    for i in l:
        
        if i in nums:
            final.append(tmp)
            tmp = []
        elif i not in ignore_list:
            tmp.append(i)
            
    final.append(tmp)
    
    np_arrs = []
    
    for val in final:
        arr = np.ones((width,)) * 48
        
        try:
            arr[:len(val)] = np.array(val)
        except:
            pass
            
        np_arrs.append(arr)
        
    return np.stack(np_arrs)

class Level:
    
    TILE_SIZE = 8
    VOID_TEXTURES = [10]
    
    def __init__(self, lvl_json):
        
        self.lvl = lvl_json
        self.attrs = self.lvl['attributes']
        self.name = self.attrs['name'].split('_')[-1]
        self.width, self.height = self.attrs['width']//Level.TILE_SIZE, self.attrs['height']//Level.TILE_SIZE
        self.xmin, self.ymin = self.attrs['x']//Level.TILE_SIZE, -self.attrs['y']//Level.TILE_SIZE - self.height
        self.xmax, self.ymax = self.xmin + self.width, self.ymin + self.height
        
        self.handle_solids()
            
        self.openings_up = [(self.xmin + x1, self.xmin+x2) for (x1,x2) in Level.find_outputs(self.img, 0, slice(0,-1))]
        self.openings_down = [(self.xmin + x1, self.xmin+x2) for (x1,x2) in Level.find_outputs(self.img, -1, slice(0,-1))]
        self.openings_left = [(self.ymin + self.height-y, self.ymin + self.height-x) for (x,y) in Level.find_outputs(self.img, slice(0,-1), 0)]
        self.openings_right = [(self.ymin + self.height-y, self.ymin + self.height-x) for (x,y) in Level.find_outputs(self.img, slice(0,-1), -1)]
        
        self.adj_up = []
        self.adj_down = []
        self.adj_left = []
        self.adj_right = []
        
        self.renderer = LevelRenderer(self.lvl)
        

        
    def consecutive(data, stepsize=1):
        return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
        
    def find_outputs(img, dim1, dim2):
    
        tmp = np.where(img[dim1, dim2] == 0)[0]

        if tmp.shape[0] == 0:
            return []

        else:
            return [(min(x),max(x)) for x in Level.consecutive(tmp)]

    def handle_solids(self):
        arrs = jmespath.search("children[?name == 'solids'].attributes.innerText" , self.lvl)
        
        if len(arrs) > 0:
            arrs = arrs[0]
            arr = decode_rle(arrs, Level.VOID_TEXTURES, self.width).astype(int)

            self.img = arr

            self.img[self.img != 48] = 1
            self.img[self.img == 48] = 0
        else:
            self.img = np.zeros((self.height, self.width))
            
    def __repr__(self):
        return f'Level(name={self.name},xmin={self.xmin},xmax={self.xmax},ymin={self.ymin},ymax={self.ymax})'
    
class LevelRenderer:
    
    TILE_SIZE = 8
    VOID_TEXTURES = [10]
    
    ID_MAP = {'badelineBoost': 2,
             'booster': 3,
             'bounceBlock': 4,
             'infiniteStar': 5,
             'bigSpinner': 6,
             'wallSpringLeft': 7,
             'wallSpringRight': 8,
             'cloud': 9,
             'coverupWall': 10,
             'crumbleBlock': 11,
             'crushBlock': 12,
             'dashBlock': 13,
             'dreamBlock': 14,
             'fallingBlock': 15,
             'fireBall': 16,
             'jumpThru': 17,
             'moveBlock': 18,
             'node': 19,
             'player': 20,
             'refill': 21,
             'seeker': 22,
             'sinkingPlatform': 23,
             'spikesDown': 24,
             'spikesLeft': 25,
             'spikesRight': 26,
             'spikesUp': 27,
             'spinner': 28,
             'spring': 29,
             'strawberry': 30,
             'zipMover': 31,
             'finish': 32,
             }
    
    RECONSTRUCT_PRIORITIES = [
                              17,
                              20,
                              32,
                              2,
                              3,
                              4,
                              5,
                              6,
                              7,
                              8,
                              31,
                              14,
                              10,
                              11,
                              12,
                              13,
                              14,
                              15,
                              9,
                              16,
                              18,
                              19,
                              20,
                              21,
                              22,
                              23,
                              24,
                              25,
                              26,
                              27,
                              28,
                              29,
                              30,
                              31]
        
    max_idx = max(ID_MAP.values())
    entity_values = range(max_idx+1)

    norm = plt.Normalize(vmin=0, vmax=max_idx)
    cm = plt.cm.nipy_spectral
        
    def __init__(self, lvl_json):
        
        self.lvl = lvl_json
        self.attrs = self.lvl['attributes']
        self.width, self.height = self.attrs['width']//LevelRenderer.TILE_SIZE, self.attrs['height']//LevelRenderer.TILE_SIZE
        self.img = None
        
        self.entities = jmespath.search("children[?name == 'entities'].children[]" , self.lvl)
        
        self.handle_solids()
        
        entity_handlers = {
            'player':self.handle_player,
            'spikesUp':self.handle_spikeup,
            'spikesLeft':self.handle_spikeleft,
            'spring':self.handle_spring,
            'zipMover':self.handle_zipmover,
            'node':self.handle_node,
            'refill':self.handle_refill,
            'booster':self.handle_booster,
            'infiniteStar':self.handle_infiniteStar,
            'cloud':self.handle_cloud,
            'bigSpinner':self.handle_bigSpinner
        }
        
        for k,v in entity_handlers.items():
            v(jmespath.search(f"[?name == '{k}']", self.entities))
        
        for entity in self.entities:
            if entity['name'] not in entity_handlers and entity['name'] in LevelRenderer.ID_MAP:
                self.generic_handler(entity)
        
    @staticmethod
    def color_to_idx(img):
    
        color_dict = {i:tuple([int(255*x) for x in LevelRenderer.cm(LevelRenderer.norm(i))]) for i in LevelRenderer.entity_values}
        rev_dict = {b:a for a,b in color_dict.items()}

        restruc = nlr.unstructured_to_structured(img).astype('O')

        return np.vectorize(rev_dict.get)(restruc)

    def render_finish(self, dim1, dim2):
        tmp = self.img.copy()
        tmp[dim1, dim2] = LevelRenderer.ID_MAP['finish']
        return LevelRenderer.cm(LevelRenderer.norm(tmp))
    
    def handle_solids(self):
        arrs = jmespath.search("children[?name == 'solids'].attributes.innerText" , self.lvl)
        
        if len(arrs) > 0:
            arrs = arrs[0]
            arr = decode_rle(arrs, LevelRenderer.VOID_TEXTURES, self.width).astype(int)

            self.img = arr

            self.img[self.img != 48] = 1
            self.img[self.img == 48] = 0
        else:
            self.img = np.zeros((self.height, self.width))

    def handle_player(self, entities):
    
        if len(entities) > 0:
            x,y = jmespath.search("attributes.[x,y]", entities[0])
            x = (x // LevelRenderer.TILE_SIZE) - 1
            y = (y // LevelRenderer.TILE_SIZE) - 2

            self.img[y:y+2, x:x+2] = LevelRenderer.ID_MAP['player']
        
    def handle_spikeup(self, entities):
        for entity in entities:
            self.generic_handler(entity, y_offset=-1)
            
    def handle_spring(self, entities):
        for entity in entities:
            self.generic_handler(entity, y_offset=-1, x_offset=-1, width_override=16)
            
    def handle_spikeleft(self, entities):
        for entity in entities:
            self.generic_handler(entity, x_offset=-1)

    def handle_zipmover(self, entities):
        for entity in entities:
            self.generic_handler(entity)
            
            self.generic_handler(entity['children'][0], width_override=16, height_override=16)
    
    def handle_node(self, entities):
        for entity in entities:
            self.generic_handler(entity, width_override=16, height_override=16) 
            
    def handle_refill(self, entities):
        for entity in entities:
            self.generic_handler(entity, width_override=16, height_override=16) 
            
    def handle_booster(self, entities):
        for entity in entities:
            self.generic_handler(entity, width_override=16, height_override=16) 
            
    def handle_infiniteStar(self, entities):
        for entity in entities:
            self.generic_handler(entity, width_override=16, height_override=16)
            
    def handle_cloud(self, entities):
        for entity in entities:
            self.generic_handler(entity, width_override=16, height_override=16) 
            
    def handle_bigSpinner(self, entities):
        for entity in entities:
            self.generic_handler(entity, width_override=16, height_override=16) 
             
    def generic_handler(self, entity, x_offset=0, y_offset=0, width_override=None, height_override=None):
    
        attrs = entity['attributes']
        x,y = attrs['x'], attrs['y']
        
        width = LevelRenderer.TILE_SIZE
        height = LevelRenderer.TILE_SIZE
        
        if 'width' in attrs:
             width = attrs['width']
                
        if 'height' in attrs:
            height = attrs['height']
            
        if width_override is not None:
            width = width_override
            
        if height_override is not None:
            height = height_override
        
        x = x // LevelRenderer.TILE_SIZE
        y = y // LevelRenderer.TILE_SIZE
        width = (width // LevelRenderer.TILE_SIZE)
        height = (height // LevelRenderer.TILE_SIZE)
        
        if x < 0:
            width = width + x
            x = 0
            
        if y < 0:
            height = height + y
            y = 0
            
        self.img[y+y_offset:y+height+y_offset, x+x_offset:x+width+x_offset] = LevelRenderer.ID_MAP[entity['name']]
        