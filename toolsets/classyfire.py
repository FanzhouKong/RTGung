import numpy as np
import requests

def get_classyfire(inchikey, return_format = 'json', gnps = True, item = 'class'):
    if gnps:
        proxy_url =  "https://gnps-classyfire.ucsd.edu"
    else:
        proxy_url = "http://classyfire.wishartlab.com"
    try:
        r = requests.get('%s/entities/%s.%s' % (proxy_url, inchikey, return_format),
                             headers={
                                 "Content-Type": "application/%s" % return_format}).json()
        return(r[item]['name'])
    except:
        return(np.NAN)
