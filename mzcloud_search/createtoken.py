import requests
# from inspection tool
def login():
    # network/healder, make sure it is post type
    url = 'https://identity.luna.public-mzcloud.cmdprod.thermofisher.com/connect/token'
    data = "refresh_token=46F2ABB8DC047A8ABF9FCE67E9EB31F99983D2FA326190A20A5CB28CB0C734D2&grant_type=refresh_token&client_id=mylibrary&client_secret=todo_move"
    # data is basiaclly only thing needs to be changed for generating bear token
    # request header
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Origin": "https://beta.mzcloud.org",
                "Referer": "https://beta.mzcloud.org/"}

    response = requests.post(url, data=data, headers=headers)
    if (response.status_code == requests.codes.ok):
        token = response.json().get('access_token')
        return token
    else:
        return None