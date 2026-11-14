import os
import json
import urllib.request
import urllib.parse
import http.cookiejar

BASE_URL = "http://127.0.0.1:5000"

cookie_jar = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))

def make_multipart_body(fields, files):
    boundary = "----WebKitFormBoundaryPhytoEchoUserUpload"
    body = bytearray()
    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode('utf-8'))
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode('utf-8'))
        body.extend(f"{value}\r\n".encode('utf-8'))
    for name, (filename, content, content_type) in files.items():
        body.extend(f"--{boundary}\r\n".encode('utf-8'))
        body.extend(f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode('utf-8'))
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode('utf-8'))
        body.extend(content)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode('utf-8'))
    return bytes(body), f"multipart/form-data; boundary={boundary}"

def request_json(url, method="GET", data=None, headers=None):
    if headers is None: headers = {}
    if data is not None and isinstance(data, dict):
        req_data = json.dumps(data).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    elif data is not None:
        req_data = data
    else:
        req_data = None
    req = urllib.request.Request(url, data=req_data, headers=headers, method=method)
    with opener.open(req) as resp:
        return resp.status, json.loads(resp.read().decode('utf-8'))

def test():
    # Login
    status, res = request_json(f"{BASE_URL}/api/auth/login", method="POST", data={"identifier": "botany", "password": "nanstack"})
    assert status == 200, f"Login failed: {res}"
    print("[PASS] Logged in as botany / nanstack")
    
    # Files uploaded by the user
    user_files = [
        'demo/dataset/raw_audio/drought_clicks/2-139748-A-15.wav',
        'demo/dataset/raw_audio/ambient/1-47709-A-16.wav',
        'demo/uploads/27ac2d9316_live_stem_recording.wav'
    ]
    
    for fpath in user_files:
        if not os.path.exists(fpath):
            print(f"Skipping {fpath} (not found)")
            continue
        fname = os.path.basename(fpath)
        with open(fpath, "rb") as f:
            content = f.read()
        body, ctype = make_multipart_body({}, {"audio": (fname, content, "audio/wav")})
        status, res = request_json(f"{BASE_URL}/api/upload", method="POST", data=body, headers={"Content-Type": ctype})
        print(f"\nUploaded {fname}:")
        print(f"  HTTP Status: {status}")
        print(f"  status: {res.get('status')}")
        print(f"  is_plant_sound: {res.get('is_plant_sound')}")
        print(f"  predicted_class: {res.get('predicted_class')}")
        print(f"  confidence: {res.get('confidence')}%")
        print(f"  clicks detected: {len(res.get('click_timestamps', []))}")
        assert status == 200 and res.get('status') == 'success' and res.get('is_plant_sound') is True, f"Failed for {fname}: {res}"

    print("\nSUCCESS! ALL USER-UPLOADED FILES NOW SUCCEED AND PREDICT PROPERLY!")

if __name__ == '__main__':
    test()
