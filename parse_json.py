import json
import os
def parse_json_file(dir_name):
    json_path=os.path.join(dir_name,'examples.json')
    with open(json_path,'r') as file:
        metadata=json.load(file)
    labels= list(metadata.keys())
    return labels,metadata


if __name__ == "__main__":
    parse_json_file('nsynth-train')