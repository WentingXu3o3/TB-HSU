This repository hosts the code of model "**TB-HSU**" and the dataset "**3DHSG**" for
## TB-HSU: Hierarchical 3D Scene Understanding with Contextual Affordances 
[Wenting Xu](https://scholar.google.com/citations?user=VKwtwKwAAAAJ&hl=en), [Viorela Ila](https://scholar.google.com/citations?user=HeaQbWsAAAAJ&hl=en), [Luping Zhou](https://scholar.google.com.au/citations?user=SgofT2MAAAAJ&hl=en), [Craig Jin](https://scholar.google.com.au/citations?user=BJ2HKOYAAAAJ&hl=en)



The concept of function and affordance is a critical aspect of 3D scene understanding and supports task-oriented objectives. In this work, we develop a model that learns to structure and vary functional affordance across a 3D hierarchical scene graph representing the spatial organization of a scene. The varying functional affordance is designed to integrate with the varying spatial context of the graph. More specifically, we develop an algorithm that learns to construct a 3D hierarchical scene graph (3DHSG) that captures the spatial organization of the scene. Starting from segmented object point clouds and object semantic labels, we develop a 3DHSG with a top node that identifies the room label, child nodes that define local spatial regions inside the room with region-specific affordances, and grand-child nodes indicating object locations and object-specific affordances. To support this work, we create a custom 3DHSG dataset that provides ground truth data for local spatial regions with region-specific affordances and also object-specific affordances for each object. We employ a Transformer Based Hierarchical Scene Understanding model (TB-HSU) to learn the 3DHSG. We use a multi-task learning framework that learns both room classification and learns to define spatial regions within the room with region-specific affordances. Our work improves on the performance of state-of-the-art baseline models and shows one approach for applying transformer models to 3D scene understanding and the generation of 3DHSGs that capture the spatial organization of a room.

## News
* 🎉 Dec 2024: TB-HSU accepted by AAAI2025

### 1. Environment install
    conda environment in TB-HSU.yml
### 2. For 3DHSG dataset
    /3DHSG/dataset/3DHSG_test.json
### 3. For results reproduction 
    Run main.py
