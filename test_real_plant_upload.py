"""
End-to-End Test for Real Plant Audio Uploads via HTTP API.
Tests:
1. Authentication
2. Real Plant Stress uploads (plant_stress_1.wav, plant_stress_10.wav, plant_stress_100.wav)
3. Irrelevant sound rejection (ESC-50 human voice / non-plant)
4. Invalid file format rejection (.txt)
"""

import os
import glob
import json
import urllib.request
import urllib.parse
import http.cookiejar

BASE_URL = "http://127.0.0.1:5000"

cookie_jar = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))

def make_multipart_body(fields, files):
    boundary = "----WebKitFormBoundaryPhytoEchoTest"
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
    if headers is None:
        headers = {}
    if data is not None and isinstance(data, dict):
        req_data = json.dumps(data).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    elif data is not None:
        req_data = data
    else:
        req_data = None
        
    req = urllib.request.Request(url, data=req_data, headers=headers, method=method)
    try:
        with opener.open(req) as resp:
            content = resp.read().decode('utf-8')
            return resp.status, json.loads(content)
    except urllib.error.HTTPError as e:
        err_content = e.read().decode('utf-8')
        try:
            return e.code, json.loads(err_content)
        except Exception:
            return e.code, {"status": "error", "raw": err_content}


def run_tests():
    print("=== Testing Real Plant Sound Uploads & Validation ===")
    
    # 1. Login
    status, res = request_json(f"{BASE_URL}/api/auth/login", method="POST", data={
        "identifier": "botany",
        "password": "nanstack"
    })
    assert status == 200 and res.get("status") == "success", f"Login failed: {res}"
    print("[PASS] Logged in successfully as botany / nanstack")

    # 2. Upload Real Plant Audio Files from the new dataset
    test_files = [
        "demo/dataset/plant_sounds/plant_stress_1.wav",
        "demo/dataset/plant_sounds/plant_stress_10.wav",
        "demo/dataset/plant_sounds/plant_stress_50.wav",
        "demo/dataset/plant_sounds/plant_stress_100.wav"
    ]
    
    for fpath in test_files:
        if not os.path.exists(fpath):
            print(f"Skipping {fpath} (not found)")
            continue
        fname = os.path.basename(fpath)
        with open(fpath, "rb") as f:
            content = f.read()
        body, ctype = make_multipart_body({}, {"audio": (fname, content, "audio/wav")})
        status, res = request_json(f"{BASE_URL}/api/upload", method="POST", data=body, headers={"Content-Type": ctype})
        
        assert status == 200, f"Upload HTTP status {status} for {fname}: {res}"
        assert res.get("status") == "success", f"Expected success for real plant sound {fname}, got {res.get('status')}: {res.get('message')}"
        assert res.get("is_plant_sound") is True, f"Expected is_plant_sound True for {fname}"
        assert res.get("predicted_class") is not None, f"Expected predicted_class for {fname}"
        print(f"[PASS] Uploaded {fname} -> {res.get('predicted_class')} (Confidence: {res.get('confidence')}%, Clicks: {len(res.get('click_timestamps', []))})")

    # 3. Test Irrelevant Audio Rejection via Upload
    irrel_files = glob.glob("demo/dataset/raw_audio/irrelevant/*.wav")[:3]
    for ipath in irrel_files:
        fname = os.path.basename(ipath)
        with open(ipath, "rb") as f:
            content = f.read()
        body, ctype = make_multipart_body({}, {"audio": (fname, content, "audio/wav")})
        status, res = request_json(f"{BASE_URL}/api/upload", method="POST", data=body, headers={"Content-Type": ctype})
        assert status == 200, f"Expected 200 for screened upload, got {status}"
        assert res.get("status") == "irrelevant_audio", f"Expected irrelevant_audio for {fname}, got {res.get('status')}"
        assert res.get("is_plant_sound") is False, f"Expected is_plant_sound False for {fname}"
        assert res.get("predicted_class") is None, f"Expected predicted_class None for {fname}"
        print(f"[PASS] Correctly screened & rejected irrelevant audio: {fname} -> Reason: {res.get('relevance_details', {}).get('reason')}")

    # 4. Test Invalid Non-Audio File Format Rejection
    txt_content = b"This is a text document, not an audio signal."
    body, ctype = make_multipart_body({}, {"audio": ("sample.txt", txt_content, "text/plain")})
    status, res = request_json(f"{BASE_URL}/api/upload", method="POST", data=body, headers={"Content-Type": ctype})
    assert status == 400, f"Expected 400 for .txt file, got {status}"
    assert res.get("status") == "invalid_format", f"Expected invalid_format, got {res.get('status')}"
    print(f"[PASS] Correctly rejected invalid file format (.txt) with HTTP 400: {res.get('message')}")

    print("\nALL REAL PLANT UPLOAD TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
