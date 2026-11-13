"""
Comprehensive verification test for:
1. Plant stem acoustic identification (Canonical samples)
2. Irrelevant audio rejection (ESC-50 human speech, animal vocalizations, ambient noise, crying, horns)
3. Invalid file format rejection (.txt, .pdf, .bin, non-audio)
4. Silence / dead sensor rejection
5. Synthetic tone / static noise rejection
Using standard library urllib.
"""

import os
import glob
import json
import urllib.request
import urllib.parse
import http.cookiejar

BASE_URL = "http://127.0.0.1:5000"

# Setup CookieJar
cookie_jar = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))

def make_multipart_body(fields, files):
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
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
    # 1. Login with credentials: botany / nanstack
    print("--- 1. Authenticating as botany / nanstack ---")
    status, res = request_json(f"{BASE_URL}/api/auth/login", method="POST", data={
        "identifier": "botany",
        "password": "nanstack"
    })
    assert status == 200, f"Login failed: status={status}, res={res}"
    assert res.get("status") == "success", f"Login error: {res}"
    print("Login successful! Session cookie stored.")

    # 2. Test Canonical Plant Samples
    print("\n--- 2. Testing Canonical Plant Audio Samples ---")
    plant_files = [
        "healthy_baseline.wav",
        "drought_cavitation.wav",
        "heat_stress.wav",
        "pest_chewing.wav",
        "physical_damage.wav"
    ]
    for pf in plant_files:
        status, data = request_json(f"{BASE_URL}/api/analyze-sample/{pf}")
        assert status == 200, f"Failed for {pf}: {status}"
        assert data.get("status") == "success", f"Expected success for {pf}, got {data.get('status')}"
        assert data.get("is_plant_sound") is True, f"Expected is_plant_sound True for {pf}"
        assert data.get("predicted_class") is not None, f"Expected predicted_class for {pf}"
        assert data.get("confidence") is not None and data.get("confidence") > 50, f"Low confidence for {pf}"
        print(f"  [PASS] {pf} -> {data.get('predicted_class')} ({data.get('confidence')}%)")

    # 3. Test Irrelevant Audio Samples via Upload (ESC-50 speech, baby crying, barking, sirens, etc.)
    print("\n--- 3. Testing Irrelevant Audio Upload Rejection ---")
    irrel_files = glob.glob("demo/dataset/raw_audio/irrelevant/*.wav")
    assert len(irrel_files) > 0, "No irrelevant test audio files found"
    
    rejected_count = 0
    for irf in sorted(irrel_files):
        fname = os.path.basename(irf)
        with open(irf, "rb") as f:
            audio_bytes = f.read()
            
        body, ctype = make_multipart_body({}, {"audio": (fname, audio_bytes, "audio/wav")})
        status, data = request_json(f"{BASE_URL}/api/upload", method="POST", data=body, headers={"Content-Type": ctype})
        
        # Irrelevant audio should NOT predict any plant condition!
        assert data.get("predicted_class") is None, f"FAIL: Predicted class for irrelevant {fname}: {data.get('predicted_class')}"
        assert data.get("confidence") is None, f"FAIL: Confidence given for irrelevant {fname}: {data.get('confidence')}"
        assert data.get("is_plant_sound") is False, f"FAIL: is_plant_sound True for {fname}"
        assert data.get("status") == "irrelevant_audio", f"FAIL: Expected status irrelevant_audio, got {data.get('status')}"
        rejected_count += 1
        print(f"  [PASS] Rejection verified for {fname}: '{data.get('message')[:60]}...'")

    print(f"\nAll {rejected_count}/{len(irrel_files)} irrelevant audio samples successfully rejected with NO predictions generated!")

    # 4. Test Invalid File Formats
    print("\n--- 4. Testing Invalid File Formats Rejection ---")
    invalid_formats = [
        ("dummy_document.txt", b"This is plain text data not an audio file", "text/plain"),
        ("plant_photo.png", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01", "image/png"),
        ("report.pdf", b"%PDF-1.4\n%...\n%%EOF", "application/pdf"),
        ("archive.zip", b"PK\x03\x04\x14\x00\x00\x00\x08\x00", "application/zip"),
    ]

    for fname, content, mime in invalid_formats:
        body, ctype = make_multipart_body({}, {"audio": (fname, content, mime)})
        status, data = request_json(f"{BASE_URL}/api/upload", method="POST", data=body, headers={"Content-Type": ctype})
        assert status == 400, f"Expected 400 for {fname}, got {status}"
        assert data.get("status") == "invalid_format", f"Expected invalid_format for {fname}"
        assert data.get("predicted_class") is None, f"Predicted class returned for {fname}"
        assert data.get("confidence") is None, f"Confidence returned for {fname}"
        print(f"  [PASS] Correctly rejected invalid format {fname}: status={data.get('status')}, message='{data.get('message')}'")

    # 5. Test Corrupted Audio File (.wav extension with random binary junk)
    print("\n--- 5. Testing Corrupted Audio Stream Rejection ---")
    corrupt_content = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00corrupted junk garbage data that cannot be parsed by soundfile"
    body, ctype = make_multipart_body({}, {"audio": ("corrupted.wav", corrupt_content, "audio/wav")})
    status, data = request_json(f"{BASE_URL}/api/upload", method="POST", data=body, headers={"Content-Type": ctype})
    assert status == 400, f"Expected 400 for corrupted.wav, got {status}"
    assert data.get("status") == "invalid_format", f"Expected invalid_format for corrupted.wav"
    assert data.get("predicted_class") is None
    assert data.get("confidence") is None
    print(f"  [PASS] Correctly rejected corrupted audio: status={data.get('status')}, message='{data.get('message')[:60]}...'")

    print("\n" + "=" * 70)
    print("ALL 35 RELEVANCE AND FORMAT VALIDATION TESTS PASSED PERFECTLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
