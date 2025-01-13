1. Download 3DHSG
3DHSG can be found in 3DHSG.json, and it follows the structure:

```
{
    "RoomID":{
        "scanID":"0a4b8ef6-a83a-21f2-8672-dce34dd0d7ca",
        "roomtype": "bedroom",
        "regions":{
            "0":{
                "region_specific_aff":"",
                "contents":[ID1,ID2,ID3],
                "region_centroid":[]
            },
            "1":{
                "region_specific_aff":"",
                "contents":[ID4,ID5,ID6],
                "region_centroid":[]
            }
            }
        }
        "objects":{
            "3": {
                "label": "tv",
                "context-specific affordances": [
                    "object-specific affordances",
                    "region-specific afforances"
                ],
                 "H3DSG_obj_relabel": "tv",
                "obj_shown_in": [...],
                "attributes": {
                ...
                },
                "segments": [
                    ...],
                ...
        }
    }
}
```
2. 3DHSG Visualization
* Firstly, follow the 3RScan downloading instructions and download the 3RScan dataset first.
* RUN open3d_label_plot.py to visualize objects' points, objectID, and label data.
