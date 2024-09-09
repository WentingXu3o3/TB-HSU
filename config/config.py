#### read config json file and load as a class 
### @ Wenting Xu 20240502
import os
import json

class GetConfig:
    def __init__(self, config_file_path):

        self.device = "" # default device

        with open(config_file_path, 'r') as jf:
            self.config = json.load(jf)

    def __getattr__(self, name):
        if name in self.config:
            return self.config.get(name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    @property
    def transformer_width(self):
        if self.cat:
            return 2 * self.embed_dim
        else:
            return self.embed_dim

    @property
    def dataset(self):
        return self.__getattr__("dataset")
    
    @property
    def embed_dim(self):
        return self.__getattr__("embed_dim")

    @property
    def cat(self):
        return self.__getattr__("cat")

    @property
    def num_objects(self):
        if self.dataset == 'ScanNet':
            if self.SE_type == 'nnEmbedding':
                if self.numLabels == 20:
                    return 62 
                    # return 54 #no wall,floor,ceiling
                elif self.numLabels == 200:
                    # return 75 # no wall,floor,ceiling
                    return 121
            else:
                return 110
        elif self.dataset == '3DHSG':
            return 77
        elif self.dataset == 'Matterport3D':
            return 230
        # return self.__getattr__("num_objects")


    @property
    def transformer_layers(self):
        return self.__getattr__("transformer_layers")

    @property
    def dropout(self):
        return self.__getattr__("dropout_rate")

    @property
    def semantic_embedding_is(self):
        return self.__getattr__("semantic_embedding_is")

    @property
    def positional_embedding_is(self):
        return self.__getattr__("positional_embedding_is")

    @property
    def print_log(self):
        return self.__getattr__("print_log")

    @property
    def pre_layer_norm(self):
        return self.__getattr__("pre_layer_norm")
    
    @property
    def pre_objs_meancentroid(self):
        return self.__getattr__("pre_objs_meancentroid")
    
    @property
    def visualize_attention_matrix(self):
        return self.__getattr__("visualize_attn_matrix")

    @property
    def relative_distance(self):
        return self.__getattr__("relative_distance")
