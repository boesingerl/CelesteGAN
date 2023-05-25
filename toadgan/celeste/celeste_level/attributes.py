"""Attributes to convert back from tensor to json"""

from typing import Optional, Any, List, TypeVar, Dict
from dataclasses import dataclass, field
from skimage import measure
from collections import ChainMap
import numpy as np
from .level import Level, LevelRenderer

import scipy
import matplotlib.pyplot as plt

from perlin_numpy import generate_perlin_noise_2d
import scipy
import cv2

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
    
def capfirst(s):
    return s[:1].upper() + s[1:]

def lowfirst(s):
    return s[:1].lower() + s[1:]

def encode_rle(img, fillvalue=71):
    replace_dict = dict({i:48 for i in range(33)})
    replace_dict[1] = fillvalue

    all_items = []

    img = np.vectorize(replace_dict.get)(img)

    for row in img:

        if (row == 48).all():
            all_items.append(10)
        else:
            
            its = []
            
            ignore = False
            
            for i,item in enumerate(row):
                if (row[i:] == 48).all():
                    its.append(10)
                    all_items.extend(its)
                    ignore = True
                    break
                else:
                    its.append(item)
                    
            if not ignore:
                its.append(10)
                all_items.extend(its)
            
    return [int(x) for x in all_items][:-1]


def map_from_tensors(levels):
    X_OFFSET = 0
    Y_OFFSET = -10

    x = 0
    y = 0

    rooms = []

    for i, lvl in enumerate(levels):

        # take highest weight tile
        idx = lvl[0].argmax(0)

        if i > 0:
            y += (idx.shape[-2] + Y_OFFSET)*LevelRenderer.TILE_SIZE

        
        room = Room.from_image(idx, x=x, y=-y, name=f'{i:03d}', lastroom=(i == len(levels)-1))
        rooms.append(room)

        x += (idx.shape[-1] + X_OFFSET)*LevelRenderer.TILE_SIZE


    # create the map json
    return {'type': "CELESTE MAP",
              'root': Map(attr_filler=Map.MapAttributes(),
        package="",
        children=[
            Levels(Levels.LevelsAttributes(),
                  children=rooms),
            Style(Style.StyleAttributes(),
                children=[
                    StyleForegrounds(attr_filler=StyleForegrounds.StyleForegroundsAttributes()),
                    StyleBackgrounds(attr_filler=StyleBackgrounds.StyleBackgroundsAttributes(),children=[
                        Parallax(attr_filler=Parallax.ParallaxAttributes(texture="purplesunset",
                                                                         scrollx=0.5,
                                                                         scrolly=0.5,
                                                                         x=0.,
                                                                         y=0.,
                                                                         speedx=4.,
                                                                         speedy=4.,
                                                                         alpha=0.5,
                                                                         blendmode="additive"
                                                                         
                                                                        )),
                        Parallax(attr_filler=Parallax.ParallaxAttributes(texture="bgs/07/bg0",
                                                                         scrollx=0.05,
                                                                        )),
                        Parallax(attr_filler=Parallax.ParallaxAttributes(texture="bgs/07/00/bg1",
                                                                         scrollx=0.5,
                                                                        )),
                        Parallax(attr_filler=Parallax.ParallaxAttributes(texture="bgs/07/00/bg2",
                                                                         scrollx=0.8,
                                                                        )),
                    ])

                ]
            ),
            Meta(attr_filler=Meta.MetaAttributes(), children=[
                MetaMode(attr_filler=MetaMode.MetaModeAttributes(), children=[
                    AudioState(attr_filler=AudioState.AudioStateAttributes())
                ]),
                CassetteModifier(attr_filler=CassetteModifier.CassetteModifierAttributes())
            ])
        ]).json()
    }

def all_annotations(cls) -> ChainMap:
    """Returns a dictionary-like ChainMap that includes annotations for all 
       attributes defined in cls or inherited from superclasses."""
    return ChainMap(*(c.__annotations__ for c in cls.__mro__ if '__annotations__' in c.__dict__) )

Lookup = TypeVar('Lookup', bound=str)
RLE = TypeVar('RLE', bound=list)
S16 = TypeVar('S16', bound=int)

class Attributes:
    
    repdict = {
        bool:'boolean',
        Lookup:'lookup',
        int:'u8',
        RLE:'rle',
        S16:'s16',
        float:'float'
     }
        
    def attr_types(self):
        return {k:Attributes.repdict.get(v) for k,v in all_annotations(type(self)).items() if v in Attributes.repdict}
    
    def attrs(self):
        return self.__dict__
    
@dataclass
class Item:
    
    attr_filler: Attributes = field(repr=False)
    
    name: Lookup = field(init=False)
    package: Lookup = None
    attributes: Dict[str, Any] = field(default_factory=dict, init=False)
    attribute_types: Dict[str, Any] = field(default_factory=dict, init=False)
    children: List[Any] = field(default_factory=list)
    
    def __post_init__(self):
        self.name = "None"
        self.attributes = self.attr_filler.attrs()
        self.attribute_types = self.attr_filler.attr_types()
        
        del self.attr_filler
        
    def json(self):
        return {k:([x.json() if hasattr(x, 'json') else x for x in v] if isinstance(v, list) else v)
                for k,v in sorted(self.__dict__.items())}

    
class Room(Item):
    
    def __post_init__(self):
        super().__post_init__()
        self.name = "level"
    
    @classmethod
    def from_image(cls, img, x=0, y=0, name='0', lastroom=False):
        
        imshape = ((max(img.shape)//4)+1)*4
        perlin = (generate_perlin_noise_2d((imshape, imshape), (4,4)) > 0).astype(int)
        perlin_scaled = perlin[:img.shape[0], :img.shape[1]]
        
        return Room(attr_filler=Room.RoomAttributes(width=img.shape[1]*TILE_SIZE,
                                                   height=img.shape[0]*TILE_SIZE,
                                                   x=x,
                                                   y=y,
                                                   name=name),
                    children=[
                        Solids.from_image(img),
                        Background.from_perlin(perlin_scaled),
                        BgDecals.from_perlin(perlin_scaled),
                        Entities(attr_filler=Entities.EntitiesAttributes(),
                                 children=Entities.from_image(img, lastroom=lastroom))
                        
                    ])
    
    @dataclass
    class RoomAttributes(Attributes):
        musicLayer4: bool = True
        x: S16 = 0
        y: S16 = 0
        c: S16 = 0
        underwater: bool = False
        dark: bool = False
        music:Lookup= "music_summit_main"
        musicLayer2:bool = True
        musicLayer3:bool = True
        cameraOffsetY:int = 0
        cameraOffsetX:int = 0
        ambienceProgress:Lookup = ""
        windPattern:Lookup = "None"
        ambience:Lookup = ""
        alt_music:Lookup = ""
        disableDownTransition:bool = False
        delayAltMusicFade:bool = False
        whisper:bool = False
        width:S16 = 320
        musicLayer1:bool= True
        height:S16=184
        musicProgress:Lookup= ""
        name:Lookup= "0"
        space:bool= False
        
class Solids(Item):
    
    def __post_init__(self):
        super().__post_init__()
        self.name = "solids"
    
    @dataclass
    class SolidsAttributes(Attributes):
        innerText: RLE
        
    @classmethod
    def from_image(cls, img, **kwargs):
        return cls(attr_filler=Solids.SolidsAttributes(innerText=encode_rle(img)), **kwargs)
        
class Background(Item):
    
    def __post_init__(self):
        super().__post_init__()
        self.name = "bg"
        
    @dataclass
    class BackgroundAttributes(Attributes):
        innerText: RLE
        
    @classmethod
    def from_perlin(cls, perlin, **kwargs):
        return cls(attr_filler=Background.BackgroundAttributes(innerText=encode_rle(perlin, fillvalue=97)), **kwargs)

class Meta(Item):
    
    def __post_init__(self):
        super().__post_init__()
        self.name = "meta"
        
    @dataclass
    class MetaAttributes(Attributes):
        TitleAccentColor: Lookup = "2f344b"
        BloomStrength: float = 1.3
        Interlude: bool = False
        Dreaming: bool = False
        TitleTextColor: Lookup = "ffffff"
        OverrideASideMeta: bool = False
        DarknessAlpha: float = 0.05
        IntroType: Lookup = "WalkInRight"
        TitleBaseColor: Lookup = "6c7c81"
        BloomBase: float = 0.3
    
class MetaMode(Item):
    
    def __post_init__(self):
        super().__post_init__()
        self.name = "mode"
        
    @dataclass
    class MetaModeAttributes(Attributes):
        HeartIsEnd: bool = True
        Inventory: Lookup = "TheSummit"
        SeekerSlowdown: bool = True
        TheoInBubble: bool = False
        IgnoreLevelAudioLayerData: bool = False
        
class AudioState(Item):
    
    def __post_init__(self):
        super().__post_init__()
        self.name = "audiostate"
        
    @dataclass
    class AudioStateAttributes(Attributes):
        Ambience: Lookup = "event:/env/amb/09_main"
        Music: Lookup = "event:/music/lvl3/intro"
        
class CassetteModifier(Item):
    
    def __post_init__(self):
        super().__post_init__()
        self.name = "cassettemodifier"
        
    @dataclass
    class CassetteModifierAttributes(Attributes):
        BeatsPerTick: int = 4
        TicksPerSwap: int = 2
        Blocks: int = 2
        BeatsMax: S16 = 256
        TempoMult: int = 1
        BeatIndexOffset: int = 0
        LeadBeats: int = 16
        OldBehavior: bool = False

        
class BgDecals(Item):
    
    entity_types = []
    
    def __post_init__(self):
        super().__post_init__()
        self.name = "bgdecals"
        
    @dataclass
    class BgDecalsAttributes(Attributes):
        pass
        
    @classmethod
    def from_perlin(cls, perlin, **kwargs):
        
        noise = (np.random.normal(size=(8,8))).astype(float)
        resized = (perlin > 0) & (cv2.resize(noise, dsize=perlin.T.shape, interpolation=cv2.INTER_AREA ) > 0)

        decals = []
        for miny, minx, height, width in EntityAttributes.extract_blobs(resized):
            decals.append(Decal(attr_filler=Decal.DecalAttributes(x=int(minx + width//2)*8,
                                                                  y=int(miny+height//2)*8,
                                                                  texture=np.random.choice(Decal.entity_types))))
            
        return cls(attr_filler=BgDecals.BgDecalsAttributes(), children=decals)
                   
class Decal(Item):
    
    entity_types = ["6-reflection/crystal_a.png",
                    "6-reflection/crystal_b.png",
                    "6-reflection/crystal_c.png", 
                    "6-reflection/crystal_shard_a.png",
                    "6-reflection/crystal_shard_b.png",
                    "6-reflection/crystal_shard_c.png",
                    "6-reflection/crystal_shard_d.png",
                    "6-reflection/crystal_shard_e.png"]
                    
    
    def __post_init__(self):
        super().__post_init__()
        self.name = "decal"
        
    @dataclass
    class DecalAttributes(Attributes):
        scaleX: S16 = 1
        scaleY: S16 = 1
        x: S16 = 0
        y: S16 = 0
        texture: Lookup = "6-reflection/crystal_a.png"
        
class Entities(Item):
    
    def __post_init__(self):
        super().__post_init__()
        self.name = "entities"
        
    @staticmethod
    def from_image(img, lastroom=False):
        items = []

        globs = globals()
        
        reverse_map = {b:a for a,b in LevelRenderer.ID_MAP.items()}

        for i in LevelRenderer.RECONSTRUCT_PRIORITIES:

            
            if i == 32 and lastroom:
                items += BlackGem.from_mask(img == i, img=img)
                
            # 0 is void, 1 is wall
            elif i > 1:
                name = reverse_map[i]

                clsname = capfirst(name)

                if clsname in globs and (img == i).sum() != 0:
                    items += globals()[clsname].from_mask(img == i, img=img)


        return items
        
    @dataclass
    class EntitiesAttributes(Attributes):
        
        def __init__(self):
            self.attrs = dict
            self.attr_types = dict
   
        
class Levels(Item):
    
    def __post_init__(self):
        super().__post_init__()
        self.name = "levels"
        
    @dataclass
    class LevelsAttributes(Attributes):
        
        def __init__(self):
            self.attrs = dict
            self.attr_types = dict
            
class Style(Item):
    
    def __post_init__(self):
        super().__post_init__()
        self.name = "Style"
        
    @dataclass
    class StyleAttributes(Attributes):
        
        def __init__(self):
            self.attrs = dict
            self.attr_types = dict
            
class StyleForegrounds(Item):
    
    def __post_init__(self):
        super().__post_init__()
        self.name = "Foregrounds"
        
    @dataclass
    class StyleForegroundsAttributes(Attributes):
        
        def __init__(self):
            self.attrs = dict
            self.attr_types = dict
            
class StyleBackgrounds(Item):
    
    def __post_init__(self):
        super().__post_init__()
        self.name = "Backgrounds"
        
    @dataclass
    class StyleBackgroundsAttributes(Attributes):
        
        def __init__(self):
            self.attrs = dict
            self.attr_types = dict
            
class Parallax(Item):
    
    def __post_init__(self):
        super().__post_init__()
        self.name = "parallax"
        
    @dataclass
    class ParallaxAttributes(Attributes):
        scrolly: S16 = 0
        scrollx: S16 = False
        
        speedx: S16 = 0
        speedy: S16 = 0
        
        loopx: bool = True
        loopy: bool = True
        
        flipx: bool = False
        flipy: bool = False
        
        x: S16 = 0
        y: S16 = 0
        
        fadex: Lookup = ""
        fadey: Lookup = ""
            
        instantIn: bool = False
        blendmode: Lookup = "alphablend"
        texture: Lookup = "bgs/07/bg0"
        notflag: Lookup = ""
        fadeIn: bool = False
        tag: Lookup = ""
        flag: Lookup = ""
        exclude: Lookup = ""
        only: Lookup = "*"
        alpha: S16 = 1
        color: Lookup = "FFFFFF"
        instantOut: bool = False

class Map(Item):
    
    def __post_init__(self):
        super().__post_init__()
        self.name = "Map"
        
    @dataclass
    class MapAttributes(Attributes):
        _package: Lookup = ""
        

TILE_SIZE = 8

@dataclass(eq=True, frozen=True)
class EntityAttributes(Attributes):
    x: S16
    y: S16
    id: int
    
    def json(self, children=None):
        return {
            'name':lowfirst(self.__class__.__name__),
            'package':None,
            'attributes':self.attrs(),
            'attribute_types':self.attr_types(),
            'children':[] if children is None else children
        }
    
    def extract_hlines(mask):

        lines = []
        for y, row in enumerate(mask):
            if row.sum() != 0:
                entities = [(min(x), max(x)-min(x)+1) for x in consecutive(np.where(row != 0)[0])]
                lines.append((y, entities))

        return lines

    def extract_vlines(mask):

        return EntityAttributes.extract_hlines(mask.T)

    def extract_blobs(mask):

        labels = measure.label(mask)

        all_maxima = []

        for i in range(1, np.unique(labels).shape[0]):

            ys, xs = np.where(labels == i)
            
            all_maxima.append((min(ys),min(xs),max(ys)-min(ys)+1,max(xs)-min(xs)+1))
        return all_maxima

    @classmethod
    def from_mask_blob(cls, mask, set_width=False, set_height=False, offset_x=0, offset_y=0, img=None, **kwargs):
        maximas = EntityAttributes.extract_blobs(mask)
        
        # make sure we don't have overlapping entities (ex: spinner inside dreamblock)
        for (y,x,height,width) in maximas:
            img[y:y+height, x:x+width] = 0
        
        return [cls(x=int((x+offset_x)*TILE_SIZE),
                    y=int((y+offset_y)*TILE_SIZE),
                    id=0,
                    **dict(dict(width=int(width*TILE_SIZE)) if set_width else dict(), **(dict(height=int(height*TILE_SIZE)) if set_height else dict())))
                    
                    for (y,x,height,width) in maximas]
    
    @classmethod
    def from_mask_hlines(cls, mask, set_width=False, offset_x=0, offset_y=0, img=None, **kwargs):
        lines = EntityAttributes.extract_hlines(mask)
        
        # make sure we don't have overlapping entities (ex: spinner inside dreamblock)
        for y,entities in lines:
            for (x,width) in entities:
                img[y:y+1, x:x+width] = 0
            
        return [cls(x=int((x+offset_x)*TILE_SIZE),
                    y=int((y+offset_y)*TILE_SIZE),
                    id=0,
                    **(dict(width=int(width*TILE_SIZE)) if set_width else {})) for y,entities in lines for (x,width) in entities]
    
    @classmethod
    def from_mask_vlines(cls, mask, set_height=False, offset_x=0, offset_y=0, img=None, **kwargs):
        lines = EntityAttributes.extract_vlines(mask)
        
        for x,entities in lines:
            for (y,height) in entities:
                img[y:y+height, x:x+1] = 0
        
        return [cls(x=int((x+offset_x)*TILE_SIZE),
                    y=int((y+offset_y)*TILE_SIZE),
                    id=0,
                    **(dict(height=int(height*TILE_SIZE)) if set_height else {})) for x,entities in lines for (y,height) in entities]
    
    @classmethod
    def from_mask(cls, mask, offset_x=0, offset_y=0, img=None, **kwargs):
        return cls.from_mask_blob(mask, set_width=False, set_height=False, offset_x=offset_x, offset_y=offset_y, img=img, **kwargs)
        
@dataclass(eq=True, frozen=True)
class Player(EntityAttributes):
    isDefaultSpawn:bool=True
    
    @classmethod
    def from_mask(cls, mask, **kwargs):
        return super().from_mask(mask, offset_x=1, offset_y=2, **kwargs)[:1]
    
@dataclass(eq=True, frozen=True)
class BadelineBoost(EntityAttributes):
    canSkip:bool=False
    lockCamera:bool=False
    finalCh9Boost:bool=False
    finalCh9GoldenBoost:bool=False
    finalCh9Dialog:bool=False
    
    @classmethod
    def from_mask(cls, mask, img=None, **kwargs):
        boosts = super().from_mask_blob(mask, img=img, **kwargs)
        
        return [boost.json(children=[Node(x=1000, y=-1000, id=0).json()]) for boost in boosts]
    
    
@dataclass(eq=True, frozen=True)
class Booster(EntityAttributes):
    red:bool=False

@dataclass(eq=True, frozen=True)
class BounceBlock(EntityAttributes):
    width:int
    height:int
    notCoreMode:bool=False
    
    @classmethod
    def from_mask(cls, mask, **kwargs):
        return super().from_mask_blob(mask, set_height=True, set_width=True, **kwargs)

@dataclass(eq=True, frozen=True)
class Bridge(EntityAttributes):
    width:int
    
    @classmethod
    def from_mask(cls, mask, **kwargs):
        return super().from_mask_hlines(mask, set_width=True, **kwargs)
    
@dataclass(eq=True, frozen=True)
class BridgeFixed(EntityAttributes):
    width:int
    
    @classmethod
    def from_mask(cls, mask, **kwargs):
        return super().from_mask_hlines(mask, set_width=True, **kwargs)
    
@dataclass(eq=True, frozen=True)
class Cassette(EntityAttributes):
    pass

@dataclass(eq=True, frozen=True)
class CassetteBlock(EntityAttributes):
    height: int 
    width: int
    tempo: int = 1
    
    @classmethod
    def from_mask(cls, mask, **kwargs):
        return super().from_mask_blob(mask, set_height=True, set_width=True, **kwargs)
    
@dataclass(eq=True, frozen=True)
class Cloud(EntityAttributes):
    fragile: bool = False
    small: bool = False
    
@dataclass(eq=True, frozen=True)
class CoverupWall(EntityAttributes):
    width: int
    height: int
    tiletype: Lookup = 'g'
    
    @classmethod
    def from_mask(cls, mask, **kwargs):
        return super().from_mask_blob(mask, set_height=True, set_width=True, **kwargs)
    
@dataclass(eq=True, frozen=True)
class CrumbleBlock(EntityAttributes):
    width: int
    texture: Lookup = 'default'
    
    @classmethod
    def from_mask(cls, mask, **kwargs):
        return super().from_mask_hlines(mask, set_width=True, **kwargs)
    
@dataclass(eq=True, frozen=True)
class CrushBlock(EntityAttributes):
    width: int
    height: int
    axes: Lookup = 'both'
    chillout: bool = False
    
    @classmethod
    def from_mask(cls, mask, **kwargs):
        return super().from_mask_blob(mask, set_height=True, set_width=True, **kwargs)
    
@dataclass(eq=True, frozen=True)
class DashBlock(EntityAttributes):
    width: int
    height: int
    permanent: bool = True
    blendin: bool = True
    canDash: bool = True
    tiletype: Lookup = 'g'
    
    @classmethod
    def from_mask(cls, mask, **kwargs):
        return super().from_mask_blob(mask, set_height=True, set_width=True, **kwargs)
    
@dataclass(eq=True, frozen=True)
class DreamBlock(EntityAttributes):
    width: int
    height: int
    below: bool = False
    fastmoving: bool = False
    
    @classmethod
    def from_mask(cls, mask, **kwargs):
        return super().from_mask_blob(mask, set_height=True, set_width=True, **kwargs)
    
@dataclass(eq=True, frozen=True)
class FallingBlock(EntityAttributes):
    width: int
    height: int
    behind: bool = False
    climbFall: bool = True
    tileType: Lookup = 'g'
    
    @classmethod
    def from_mask(cls, mask, **kwargs):
        return super().from_mask_blob(mask, set_height=True, set_width=True, **kwargs)
    
@dataclass(eq=True, frozen=True)
class FireBall(EntityAttributes):
    amount: int = 3
    offset: int = 0
    notCoreMode: bool = False
    speed: int = 1
    
@dataclass(eq=True, frozen=True)
class JumpThru(EntityAttributes):
    width: int
    surfaceIndex: S16 = -1
    texture: Lookup = 'reflection'
    
    @classmethod
    def from_mask(cls, mask, **kwargs):
        return super().from_mask_hlines(mask, set_width=True, **kwargs)

@dataclass(eq=True, frozen=True)
class MoveBlock(EntityAttributes):
    width: int
    height: int
    canSteer: bool = False
    direction: Lookup = "Right"
    fast: bool = False
    
    @classmethod
    def from_mask(cls, mask, **kwargs):
        return super().from_mask_blob(mask, set_height=True, set_width=True, **kwargs)
    
@dataclass(eq=True, frozen=True)
class Node(EntityAttributes):
    
    @classmethod
    def from_mask(cls, mask, **kwargs):
        return []

@dataclass(eq=True, frozen=True)
class Refill(EntityAttributes):
    twoDash: bool = False
    oneUse: bool = False
    
@dataclass(eq=True, frozen=True)
class Seeker(EntityAttributes):
    pass

@dataclass(eq=True, frozen=True)
class SinkingPlatform(EntityAttributes):
    width: int
    texture: Lookup = 'cliffside'
    
    @classmethod
    def from_mask(cls, mask, **kwargs):
        return super().from_mask_hlines(mask, set_width=True, **kwargs)
    
@dataclass(eq=True, frozen=True)
class SpikesDown(EntityAttributes):
    width: int
    type: Lookup = 'reflection'
    
    @classmethod
    def from_mask(cls, mask, **kwargs):
        return super().from_mask_hlines(mask, set_width=True, **kwargs)

@dataclass(eq=True, frozen=True)
class SpikesUp(EntityAttributes):
    width: int
    type: Lookup = 'reflection'
    
    @classmethod
    def from_mask(cls, mask, **kwargs):
        return super().from_mask_hlines(mask, set_width=True, offset_y=1, **kwargs)
    

@dataclass(eq=True, frozen=True)
class SpikesLeft(EntityAttributes):
    height: int
    type: Lookup = 'reflection'
    
    @classmethod
    def from_mask(cls, mask, **kwargs):
        return super().from_mask_vlines(mask, set_height=True, offset_x=1, **kwargs)

@dataclass(eq=True, frozen=True)
class SpikesRight(EntityAttributes):
    height: int
    type: Lookup = 'reflection'
    
    @classmethod
    def from_mask(cls, mask, **kwargs):
        return super().from_mask_vlines(mask, set_height=True, **kwargs)

@dataclass(eq=True, frozen=True)
class Spinner(EntityAttributes):
    dust: bool = False
    color: Lookup = 'blue'
    attachToSolid: bool = False

@dataclass(eq=True, frozen=True)
class Spring(EntityAttributes):
    playerCanUse: bool = True
    
    @classmethod
    def from_mask(cls, mask, **kwargs):
        return super().from_mask(mask, offset_x=1, offset_y=1, **kwargs)
    
@dataclass(eq=True, frozen=True)
class Strawberry(EntityAttributes):
    order: S16 = -1
    winged: bool = False
    checkpointID: S16 = -1
    moon: bool = False
    
@dataclass(eq=True, frozen=True)
class ZipMover(EntityAttributes):
    width: int
    height: int
    theme: Lookup = 'Normal'
    
    @classmethod
    def from_mask(cls, mask, img=None, **kwargs):
        # return super().from_mask_blob(mask, set_height=True, set_width=True)
        
        zips = super().from_mask_blob(mask, set_height=True, set_width=True, img=img, **kwargs)
        
        nodes = Node.from_mask_blob(img == 19, img=img, **kwargs)
        
        allzips = []

        while len(nodes) > 0 and len(zips) > 0:

            noded = [(n.x, n.y) for n in nodes]
            ziped = [(z.x, z.y) for z in zips]

            dist = scipy.spatial.distance.cdist(noded, ziped)
            minind = dist.argmin()
            nind, zind = np.unravel_index(minind, dist.shape)

            allzips.append(zips[zind].json(children=[nodes[nind].json()]))

            del nodes[nind]
            del zips[zind]
            
        return allzips
    
@dataclass(eq=True, frozen=True)
class Finish(EntityAttributes):
    height: S16
    width: S16 = 16
    originX: S16 = 0
    originY: S16 = 0
    flag: Lookup = ""
    rotation: S16 = 90
    
    
    def json(self, children=None):
        return {
            'name':'lightbeam',
            'package':None,
            'attributes':self.attrs(),
            'attribute_types':self.attr_types(),
            'children':[] if children is None else children
        }
    
    @classmethod
    def from_mask(cls, mask, offset_x=1, offset_y=3.5, img=None, **kwargs):
        
        maximas = EntityAttributes.extract_blobs(mask)
        return [cls(x=int((x+(width if width >= 2 else 1))*TILE_SIZE),
                    y=int((y+offset_y)*TILE_SIZE),
                    id=0,
                    **dict(dict(width=int(height*TILE_SIZE)), **dict(height=int(2*TILE_SIZE))))
                    
                    for (y,x,height,width) in maximas]

@dataclass(eq=True, frozen=True)
class BlackGem(EntityAttributes):
    
    @classmethod
    def from_mask(cls, mask, img=None, **kwargs):
        
        maximas = EntityAttributes.extract_blobs(mask)
        return [cls(x=int((x+width//2)*TILE_SIZE),
                    y=int((y+height//2)*TILE_SIZE),
                    id=0)
                    for (y,x,height,width) in maximas]