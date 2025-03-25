import ssl
import socket
import sys
import os
import argparse
import base64
import ast
import time
import datetime
import tarfile

import xml.etree.ElementTree as etree

try:
    from httplib import HTTPSConnection
    import httplib
    import urllib
    import urllib.error
except:
    from http.client import HTTPSConnection
    import http.client as httplib
    import urllib.error
    import urllib.request


try:
    sslcontext = ssl._create_unverified_context()
except:
    sslcontext = None


def check_content_length(rec):
    if 'Content-Length' in rec:
        return True
    elif 'content-length' in rec:
        return True
    elif 'Content-Length' in rec:
        return True
    return False


def get_content_length(rec):
    if 'Content-Length' in rec:
        return int(rec['Content-Length'])
    elif 'content-length' in rec:
        return int(rec['content-length'])
    elif 'Content-Length' in rec:
        return int(rec['Content-Length'])
    return -1


def getauthorization(username, password):
    return "Basic {}".format(base64.b64encode('{}:{}'.format(username, password).encode()).decode("ascii"))

class HTTPSConnectionV3(HTTPSConnection):
    def __init__(self, *args, **kwargs):
        httplib.HTTPSConnection.__init__(self, *args, **kwargs)

    def connect(self):
        sock = socket.create_connection((self.host, self.port), self.timeout)
        if self._tunnel_host:
            self.sock = sock
            self._tunnel()
        ssl_version_list = [ssl.PROTOCOL_SSLv2,ssl.PROTOCOL_SSLv3,ssl.PROTOCOL_SSLv23,
                            ssl.PROTOCOL_TLSv1]

        for ssl_i in ssl_version_list:
            try:
                self.sock = ssl.wrap_socket(sock, self.key_file, self.cert_file, cert_reqs=ssl.CERT_NONE, ssl_version=ssl.PROTOCOL_SSLv3)
                break
            except ssl.SSLError as e:
                print("Failed:"+ssl._PROTOCOL_NAMES[ssl_i])
#            self.sock = ssl.wrap_socket(sock, self.key_file, self.cert_file, cert_reqs=ssl.CERT_NONE, ssl_version=ssl.PROTOCOL_SSLv23)
#            self.sock = ssl.wrap_socket(sock, self.key_file, self.cert_file, ssl_version=ssl.PROTOCOL_SSLv23)
#            self.sock = ssl.wrap_socket(sock, self.key_file, self.cert_file, ssl_version=ssl.PROTOCOL_SSLv2)


BASE_EAS_URL="https://eas-dps-rest-ops.esac.esa.int/REST?class_name="
#BASE_EAS_URL="https://eas-dps-rest-rep1.esac.esa.int/REST?class_name="

BASE_DSS_URL="https://euclidsoc.esac.esa.int/"
BASE_DSS_HOST="euclidsoc.esac.esa.int"
BASE_DSS_PORT=443
buffer_size=16*1024

def geturl(inpstring):
    url = ''
    jobstatus = ''
    try:
       retdic=ast.literal_eval(inpstring)
#       print(retdic)
       if 'url' in retdic:
           url = retdic['url']
       if 'status' in retdic:
           jobstatus = retdic['status']
#       print(url,jobstatus)
    except:
       print("Can not decode string: %s" % inpstring)
       exit()
    return url, jobstatus


def wrap_urlopen(request, max_tries=3):
    """ """
    response = None
    for loop in range(max_tries):
        try:
            response = urllib.request.urlopen(request)
        except (ConnectionResetError, urllib.error.HTTPError):
            time.sleep(1)
            continue
    if response is None:
        raise ConnectionResetError
    return response


def checkasyjob(url, auth):
    spinner = ["✶", "✸", "✹", "✺", "✹", "✷"]
    finished = False
    t0 = time.time()
    counter = 0
    jobstatus = "wait..."
    while True:
        time.sleep(0.1)
        dt = time.time() - t0
        sys.stdout.write(f"\r{spinner[counter%len(spinner)]} Elapsed {dt:3.0f} sec, Query status: {jobstatus}               ")
        sys.stdout.flush()
        counter += 1
        if counter > 50:
            counter = 0
            try:
                request = urllib.request.Request(url)
                request.add_header("Authorization", auth)
                easResponse = wrap_urlopen(request)
                jobresponse = easResponse.read().decode()
                url, jobstatus = geturl(jobresponse)
            except ConnectionResetError:
                continue
            if jobstatus=='FINISHED':
                finished = True
                break
            elif jobstatus=='ERROR':
                break
    sys.stdout.write(f"\r{spinner[counter%len(spinner)]} Elapsed {dt:3.0f} sec, Query status: {jobstatus}\n")
    return url, finished


def getMetadataXml(base_url, product_type, product_query, project, username, password):
  product_query = base_url + product_type + "&" + product_query + "&make_asy=True&file_format=TGZ&PROJECT=" + project
#   print(product_query)
  print("Query submitted at %s" % datetime.datetime.now())
  request = urllib.request.Request(product_query)
  auth = getauthorization(username, password)
  request.add_header("Authorization", auth)
  easResponse = wrap_urlopen(request)
  jobresponse = easResponse.read().decode()
  url, jobstatus = geturl(jobresponse)
  print(url)
  url, finished = checkasyjob(url, auth)
  print("Job finished on server side at %s" % datetime.datetime.now())
  request = urllib.request.Request(url)
  request.add_header("Authorization", auth)
  easResponse = wrap_urlopen(request)
  timestamp = f'{datetime.datetime.now():%Y-%m-%dT%H:%M:%S.%f}'
  output_tgz = product_type + timestamp + '.tgz'
  with open(output_tgz, "wb") as f_out:
      f_out.write(easResponse.read())
  cip = 0
  ret_p = []
  if tarfile.is_tarfile(output_tgz):
      tarxml = tarfile.open(output_tgz, 'r:gz')
      for i_file in tarxml.getmembers():
          i_f = tarxml.extractfile(i_file)
          if i_f:
              i_f_content = i_f.read().decode()
              if len(i_f_content.strip()) > 0:
                  ret_p.append(i_f_content)
                  cip = cip +1
  else:
      errorfile = f'ERROR-{timestamp}'
      os.rename(output_tgz, errorfile)
      print(f'Error in executing query, see {errorfile}')
      return []
  print("Data products metadata retrieved at %s" % datetime.datetime.now())
#  print(productList)
  # Workaround for the EAS response, when a list of products is provided
#  productList = productList.replace('<?xml version="1.0" encoding="UTF-8"?>', '<?xml version="1.0" encoding="UTF-8"?><dummyRoot>') + "</dummyRoot>"
#  root_elem = etree.fromstring(productList)
  print("Found %d data products" % cip)
  return ret_p


def downloadDssFile(base_url, fname, username=None, password=None, retry=0, max_retries=3, count=0):
  #sys.stdout.write(f"\r{count} {fname:50s}                                             ")
  #sys.stdout.flush()
  if retry > max_retries:
      print(f"{fname}: download tried {retry} times and failed. :(")
      return
#   if retry > 0:
    #   print(f"Trying to download {fname} again. This is attempt {retry} of {max_retries}.")
  headers = {}
  if username and password:
      headers['Authorization'] = 'Basic %s' % (base64.b64encode(b"%s:%s" % (username.encode('utf-8'), password.encode('utf-8'))).decode('utf-8'))
  headers['pragma']='DSSGET'
  fileurl = base_url + fname
  if sslcontext:
      conn = HTTPSConnection(BASE_DSS_HOST, BASE_DSS_PORT, context=sslcontext)
  else:
      conn = HTTPSConnection(BASE_DSS_HOST, BASE_DSS_PORT)
  conn.request('GET', '/'+fname, headers=headers)
  response = conn.getresponse()
  recvheader = {}
  for k, v in dict(response.getheaders()).items():
      recvheader[k.lower()] = v
#  response = requests.get(fileurl, auth=(username, password))
  failed = True
  if response.status == 200:
      if check_content_length(recvheader):
          try:
              total_length = get_content_length(recvheader)
              size_mb = total_length / 1024**2
              with open(fname, "wb") as f:
                  if total_length is None:
                      f.write(response.content)
                  else:
                      dl = 0.0
                      dlc = 0.0
                      data = response.read(buffer_size)
                      while data:
                          dlc = len(data)
                          dl = dl + dlc
                          f.write(data)
                          done = 100*dl/total_length
                          sys.stdout.write(f"\rDownloading {count} {fname[-80:]:80s} [{size_mb:3.1f}MB] {done:3.1f}%")
                          sys.stdout.flush()
#            print(total_length,dlc,dl)
                          data = response.read(buffer_size)
                      if dl < total_length:
                           sys.stdout.write("Wrong size for file %s - need %d, got %d\n" % (fname, total_length, dl))
                #   sys.stdout.write("\n")
              failed = False
          except Exception as e:
              if os.path.isfile(fname):
                  os.remove(fname)
              print("Can't write file %s - error %s" % (fname, str(e)))
  elif response.status == 403:
    #   sys.stdout.write("Error 403\n")
    pass
  elif response.status == 404:
      reason = ''
      if hasattr(response,'reason'):
          reason = response.reason
      out_message = 'File %s not found: %s\n' % (fname, reason)
      sys.stdout.write(out_message)
  else:
      reason = ''
      if hasattr(response,'reason'):
          reason = response.reason
      out_message = 'File %s can not be downloaded: %s\n' % (fname, reason)
      sys.stdout.write(out_message)
  conn.close()
  del conn
  if failed:
      downloadDssFile(base_url, fname, username, password, retry+1, max_retries, count=count)


def saveMetaAndData(products, username=None, password=None, product_type='UNKNOWN'):
  count = 0
  for p in products:
    #findProductId = etree.XPath("//ProductId")
    #findFiles = etree.XPath("//FileName")

    root = etree.XML(p)
    ptype_node = root.find(".//ProductType")
    pid_node = root.find(".//ProductId")
    if ptype_node:
        ptype = ptype_node.text
    else:
        ptype = product_type
    if hasattr(pid_node, 'text') and pid_node.text:
        pid = pid_node.text
    else:
        id_node = root.find(".//Id")
        if id_node:
            pid = id_node.text
        else:
            pid = str(count)
    pfile = ptype[0].upper() + ptype[1:] + '__' + pid + ".xml"
    print("Saving " + pfile)
    with open(pfile,'w') as f:
      f.write(p)

    files = [f.text for f in root.findall(".//FileName")]
    for f in files:
      if os.path.isfile(f):
        print("File %s already exists locally. Skipping its download" % (f))
      else:
        print("Start retrieving of " + f + " at " +str(datetime.datetime.now())+" :")
        downloadDssFile(BASE_DSS_URL, f, username, password, count=count)
        print("Finished retrieving of " + f + " at " +str(datetime.datetime.now()))
    count = count + 1


def load_password_file(path):
    """ """
    with open(path) as inp:
        s = inp.read()
    try:
        username, password = bytes.fromhex(s).decode('utf-8').split("\n")[:2]
    except ValueError:
        username, password = s.split("\n")[:2]
    return username, password


def download(filename, dir=".", username=None, password=None, password_file=None):
    """
    """
    if password_file is not None:
        username, password = load_password_file(password_file)
    downloadDssFile(BASE_DSS_URL, filename, username, password)
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    os.rename(filename, os.path.join(dir, filename))


if __name__ == '__main__':

    FIELD_ID = ['52926','53401','53402','53403','53876','53877','53878','54348','54349']

    parser = argparse.ArgumentParser()
    parser.add_argument('--username', help='Cosmos or EAS username', required=True)
    parser.add_argument('--password', help='user password',)
    parser.add_argument('--project', help='EAS project to query', default='EUCLID')
    parser.add_argument('--data_product', help='Data product type name, e.g. DpdMerFinalCatalog', required=True)
    parser.add_argument('--query', required=True, help='Product query string, e.g. \n'
                                    'Header.ProductId.ObjectId=like*EUC_MER_PPO-TILE*_SC3-PLAN-2-PPO-*-SDC-IT-RUN0-0-final_catalog-0')

    args = parser.parse_args()

    username = args.username
    password = args.password

    if password and os.path.isfile(password):
        with open(password) as f:
            password = f.read().replace("\n", "").strip()

    if username and not password:
        import getpass
        password = getpass.getpass('Type password for %s: ' % username)

    products = getMetadataXml(BASE_EAS_URL, args.data_product, args.query, args.project, username, password)

    saveMetaAndData(products, username, password, args.data_product)


