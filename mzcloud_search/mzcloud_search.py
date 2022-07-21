import mzcloud_search.createtoken as ct
import requests
import base64


class MzCloudFinder:
    uploadUrl = "https://beta.mzcloud.org/api/file/telerikPush"
    filePayloadUrl = "https://beta.mzcloud.org/api/file"
    identityUrl = "https://beta.mzcloud.org/api/searchv2/spectrum/precusor"
    similarityUrl = "https://beta.mzcloud.org/api/searchv2/spectrum/highest"
    accessToken = ct.login()

    def __init__(self):
        pass

    def upload_file(self,path_to_file):
        filename = path_to_file.split('/')[-1].strip()
        # path = '/'.join(sys.argv[1].split('/')[0:-1])
        files = {'files': (filename, open(path_to_file, 'rb'), 'application/octet-stream')}
        headers = {'Authorization': 'Bearer %s' % self.accessToken,"Origin": "https://beta.mzcloud.org",
                "Referer": "https://beta.mzcloud.org/", 'filename': '%s' % base64.b64encode(filename.encode()).decode('utf-8')}
        r = requests.post(self.uploadUrl, files = files, headers = headers)
        # file name can be changed to variable: AU101802.msp
        if (r.status_code == requests.codes.ok):
            return r.json().get('fileKey')

    def create_payload(self,path_to_file):
        key = self.upload_file(path_to_file)
        payloadUrl = self.filePayloadUrl + '/%s/asSpectrumDto' % key
        print(payloadUrl)
        headers = {'Authorization': 'Bearer %s' % self.accessToken,
                "Referer": "https://beta.mzcloud.org/dataviewer"}
        r = requests.get(payloadUrl, headers = headers)
        if (r.status_code == requests.codes.ok):
            return r.json()

    def get_results(self, spectrum, polarity):
        headers = {'Authorization': 'Bearer %s' % self.accessToken, 'Origin': 'https://beta.mzcloud.org',
                "Referer": "https://beta.mzcloud.org/dataviewer", 'Content-Type': 'application/json'}
        data = {
                "libraries": [
                    "vvsC9HOzAl",
                    "hPqDettnTP"],
                "polarity": polarity,
                "scoringMethod": 2,
                # nist is nist, cosine is dot product, highreschem is denver
                "searchType": "ms2",
                "scoringThreshold": 75,
                "matchFactorAlgorithm": "Identity",
                "matchActType": False,
                "matchEnergy": False,
                "energyTolerance": 30,
                "metadataFilter": {
                    "boolConditions": [],
                    "textConditions": [],
                    "rangeConditions": [],
                    "rtConditions": []
                },
                "spectrum": spectrum
        }
        # data is searching parame  ter
        r = requests.post(self.identityUrl, headers = headers, json = data)
        print(r.status_code)
        print(r.text)
        if (r.status_code == requests.codes.ok):
            return r.json()
