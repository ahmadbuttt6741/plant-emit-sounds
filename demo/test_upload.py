import urllib.request
import json

with open('demo/static/samples/heat_stress.wav', 'rb') as f:
    file_bytes = f.read()

boundary = 'WebKitFormBoundaryDemo12345'
header = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="audio"; filename="test_heat_upload.wav"\r\n'
    f"Content-Type: audio/wav\r\n\r\n"
).encode('latin1')
footer = f"\r\n--{boundary}--\r\n".encode('latin1')
body = header + file_bytes + footer

req = urllib.request.Request('http://127.0.0.1:5000/api/upload', data=body)
req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')

resp = urllib.request.urlopen(req)
res = json.loads(resp.read().decode('utf-8'))
print("Upload Status:", res.get("status"))
print("File Analyzed:", res.get("filename"))
print("Prediction:", res.get("predicted_class"))
print("Confidence:", str(res.get("confidence")) + "%")
print("Clicks detected:", res.get("metrics", {}).get("cavitation_clicks"))
