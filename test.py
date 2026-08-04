# -*- coding: utf-8 -*-
# 替换 YOUR_API_KEY 为您在控制台-数据管理的真实密钥

import http.client, urllib, json
conn = http.client.HTTPSConnection('apis.tianapi.com')  #接口域名
params = urllib.parse.urlencode({'key':'7d36755f55ea230eecd1d9892bf74d1a','num':'10','word':'马市'})
headers = {'Content-type':'application/x-www-form-urlencoded'}
conn.request('POST','/shares/index',params,headers)
tianapi = conn.getresponse()
result = tianapi.read()
data = result.decode('utf-8')
dict_data = json.loads(data)
print(dict_data)