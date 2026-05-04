import mlcroissant as mlc
import json

with open('HRGC_P_croissant_v3.json', 'r') as f:
    jsonld = json.load(f)

ds = mlc.Dataset(jsonld=jsonld)
print('Errors:', ds.metadata.issues.errors)
print('Warnings:', ds.metadata.issues.warnings)