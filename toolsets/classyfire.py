import numpy as np
import requests

def get_classyfire(identifier,type = "smiles", return_format = 'json', gnps = True, item = 'class'):
    if gnps:
        proxy_url =  "https://gnps-classyfire.ucsd.edu"
        # proxy_url = "https://gnps-structure.ucsd.edu/classyfire?smiles=<smiles string>"
    else:
        proxy_url = "http://classyfire.wishartlab.com"
    try:
        if type == "smiles":
            r = requests.get('https://gnps-structure.ucsd.edu/classyfire?smiles=%s' % (identifier)).json()

            return(r[item]['name'])
        elif type == "inchikey":
            r = requests.get('%s/entities/%s.%s' % (proxy_url, identifier, return_format),
                                 headers={
                                     "Content-Type": "application/%s" % return_format}).json()

            return(r[item]['name'])
        else:
            print("the input type is wrong, please try again")
            return(np.NAN)
    except:
        # print('something went wrong in searching process')
        return(np.NAN)
