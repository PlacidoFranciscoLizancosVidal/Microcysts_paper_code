import json, pickle

def saveObj(obj, name ):
    with open(str(name) + '.json', 'w') as f:
        json.dump(obj, f)

        
def loadObj(name ):
    with open(str(name) + '.json', 'r') as f:
        return json.load(f)
