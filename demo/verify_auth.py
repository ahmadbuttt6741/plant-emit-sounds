"""
Comprehensive Verification Suite for PhytoEcho AI Landing Page & Authentication.
Tests:
1. Public Landing page delivery & content.
2. Route protection for unauthenticated users (/dashboard -> redirect).
3. API route protection for unauthenticated calls (returns 401).
4. Invalid credential rejection & error messaging.
5. Valid credential authentication & session establishment.
6. Authenticated access to /dashboard (renders user name/role & telemetry).
7. Authenticated access to protected ML APIs (/api/model-metrics & /api/analyze-sample).
8. Authenticated root access (auto-redirects to /dashboard).
9. Session logout (clears session and redirects to landing page).
10. Password reset flow (generates verification code & resets password).
"""

import sys
import json
import urllib.request
import urllib.error
import http.cookiejar

BASE_URL = "http://127.0.0.1:5000"

def run_tests():
    # Setup cookie jar to track session cookies
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(cookie_jar),
        # Do not automatically follow redirects so we can inspect 302 status codes
    )

    class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
        def http_error_302(self, req, fp, code, msg, headers):
            return fp
        http_error_301 = http_error_302
        http_error_303 = http_error_302
        http_error_307 = http_error_302

    no_redirect_opener = urllib.request.build_opener(
        NoRedirectHandler(),
        urllib.request.HTTPCookieProcessor(cookie_jar)
    )

    print("=" * 70)
    print("PHYTOECHO AI — AUTHENTICATION & LANDING PAGE TEST SUITE")
    print("=" * 70)

    # ----------------------------------------------------
    # TEST 1: Public Landing Page
    # ----------------------------------------------------
    print("\n[TEST 1] GET / (Unauthenticated visitor)")
    req = urllib.request.Request(f"{BASE_URL}/")
    with opener.open(req) as resp:
        html = resp.read().decode('utf-8')
        assert resp.status == 200, f"Expected 200 OK, got {resp.status}"
        assert "Welcome Back" in html, "Landing page missing 'Welcome Back' login panel"
        assert "Listen to" in html, "Landing page missing introduction heading"
        assert "nanstack.tech" in html, "Landing page missing Nanstack branding"
        assert "Thirst & Drought Alerts" in html or "Plant Acoustic" in html, "Landing page missing feature cards"
        print("  -> PASSED: Landing page rendered with 200 OK, branding, and login panel.")

    # ----------------------------------------------------
    # TEST 2: Protected /dashboard Route Blocks Unauthenticated
    # ----------------------------------------------------
    print("\n[TEST 2] GET /dashboard (Direct access without session)")
    resp = no_redirect_opener.open(f"{BASE_URL}/dashboard")
    assert resp.status == 302, f"Expected 302 Redirect, got {resp.status}"
    location = resp.headers.get('Location')
    print(f"  -> Redirect Location: {location}")
    assert "/" in location or "/login" in location, "Did not redirect to landing/login"
    print("  -> PASSED: Unauthenticated user successfully blocked and redirected.")

    # ----------------------------------------------------
    # TEST 3: Protected API Routes Block Unauthenticated
    # ----------------------------------------------------
    print("\n[TEST 3] GET /api/model-metrics (Unauthenticated API call)")
    try:
        urllib.request.urlopen(f"{BASE_URL}/api/model-metrics")
        assert False, "API call should have failed with 401"
    except urllib.error.HTTPError as e:
        assert e.code == 401, f"Expected 401 Unauthorized, got {e.code}"
        err_body = json.loads(e.read().decode('utf-8'))
        assert err_body.get('status') == 'error', "Error status missing"
        print(f"  -> PASSED: Rejected with 401: {err_body.get('message')}")

    # ----------------------------------------------------
    # TEST 4: Invalid Login Credentials
    # ----------------------------------------------------
    print("\n[TEST 4] POST /api/auth/login (Invalid credentials)")
    bad_payload = json.dumps({
        "identifier": "botany",
        "password": "WrongPassword999"
    }).encode('utf-8')
    bad_req = urllib.request.Request(
        f"{BASE_URL}/api/auth/login",
        data=bad_payload,
        headers={"Content-Type": "application/json"}
    )
    try:
        opener.open(bad_req)
        assert False, "Login should have failed"
    except urllib.error.HTTPError as e:
        assert e.code == 401, f"Expected 401, got {e.code}"
        res = json.loads(e.read().decode('utf-8'))
        print(f"  -> PASSED: Properly rejected invalid login: {res.get('message')}")

    # ----------------------------------------------------
    # TEST 5: Valid Login Credentials (botany / nanstack)
    # ----------------------------------------------------
    print("\n[TEST 5] POST /api/auth/login (Valid credentials: botany / nanstack)")
    good_payload = json.dumps({
        "identifier": "botany",
        "password": "nanstack",
        "remember_me": True
    }).encode('utf-8')
    good_req = urllib.request.Request(
        f"{BASE_URL}/api/auth/login",
        data=good_payload,
        headers={"Content-Type": "application/json"}
    )
    with opener.open(good_req) as resp:
        assert resp.status == 200, f"Expected 200 OK, got {resp.status}"
        data = json.loads(resp.read().decode('utf-8'))
        assert data.get('status') == 'success', "Login response status not success"
        print(f"  -> Authenticated as: {data.get('user', {}).get('name')} ({data.get('user', {}).get('role')})")
        print(f"  -> Redirect Target: {data.get('redirect')}")
        print("  -> PASSED: Successful authentication and session cookie created.")

    # ----------------------------------------------------
    # TEST 6: Authenticated Access to /dashboard
    # ----------------------------------------------------
    print("\n[TEST 6] GET /dashboard (With authenticated session cookie)")
    with opener.open(f"{BASE_URL}/dashboard") as resp:
        dash_html = resp.read().decode('utf-8')
        assert resp.status == 200, f"Expected 200 OK, got {resp.status}"
        assert "Sign Out" in dash_html, "Dashboard missing Sign Out button"
        assert "Botany Administrator" in dash_html or "Lead Bio-Acoustic" in dash_html, "Dashboard missing user role info"
        assert "PHYTOECHO" in dash_html.upper(), "Dashboard missing PhytoEcho title"
        print("  -> PASSED: Protected dashboard accessed with active session and user badge.")

    # ----------------------------------------------------
    # TEST 7: Authenticated Protected API Access
    # ----------------------------------------------------
    print("\n[TEST 7] GET /api/model-metrics & /api/analyze-sample (Authenticated)")
    with opener.open(f"{BASE_URL}/api/model-metrics") as resp:
        assert resp.status == 200, "Metrics API failed"
        metrics = json.loads(resp.read().decode('utf-8'))
        assert metrics.get('accuracy') is not None, "Metrics missing accuracy"
        print(f"  -> Model Accuracy: {metrics.get('accuracy')}% across {metrics.get('total_samples')} samples")

    with opener.open(f"{BASE_URL}/api/analyze-sample/drought_cavitation.wav") as resp:
        assert resp.status == 200, "Sample analysis failed"
        analysis = json.loads(resp.read().decode('utf-8'))
        print(f"  -> Sample Prediction: {analysis.get('predicted_class')} ({analysis.get('confidence')}%)")
        print("  -> PASSED: Protected APIs functional for authenticated user.")

    # ----------------------------------------------------
    # TEST 8: Authenticated User visiting / auto-redirects
    # ----------------------------------------------------
    print("\n[TEST 8] GET / (Authenticated user visiting root)")
    resp = no_redirect_opener.open(f"{BASE_URL}/")
    assert resp.status == 302, f"Expected 302 redirect, got {resp.status}"
    location = resp.headers.get('Location')
    print(f"  -> Auto-redirect Location: {location}")
    assert "/dashboard" in location, "Did not redirect to /dashboard"
    print("  -> PASSED: Authenticated user is automatically directed to /dashboard.")

    # ----------------------------------------------------
    # TEST 9: Forgot Password & Verification Code Flow
    # ----------------------------------------------------
    print("\n[TEST 9] Forgot Password & Reset Flow")
    fp_payload = json.dumps({"email": "researcher@nanstack.tech"}).encode('utf-8')
    fp_req = urllib.request.Request(
        f"{BASE_URL}/api/auth/forgot-password",
        data=fp_payload,
        headers={"Content-Type": "application/json"}
    )
    with opener.open(fp_req) as resp:
        fp_res = json.loads(resp.read().decode('utf-8'))
        assert fp_res.get('status') == 'success', "Forgot password failed"
        code = fp_res.get('code')
        print(f"  -> Generated Verification Code: {code}")

    # Reset password
    reset_payload = json.dumps({
        "email": "researcher@nanstack.tech",
        "code": code,
        "new_password": "NewResearcherPass@456"
    }).encode('utf-8')
    reset_req = urllib.request.Request(
        f"{BASE_URL}/api/auth/reset-password",
        data=reset_payload,
        headers={"Content-Type": "application/json"}
    )
    with opener.open(reset_req) as resp:
        reset_res = json.loads(resp.read().decode('utf-8'))
        assert reset_res.get('status') == 'success', "Password reset failed"
        print(f"  -> Reset Message: {reset_res.get('message')}")

    # Verify login with new password
    new_login_payload = json.dumps({
        "identifier": "researcher@nanstack.tech",
        "password": "NewResearcherPass@456"
    }).encode('utf-8')
    new_login_req = urllib.request.Request(
        f"{BASE_URL}/api/auth/login",
        data=new_login_payload,
        headers={"Content-Type": "application/json"}
    )
    with opener.open(new_login_req) as resp:
        assert resp.status == 200, "Login with new password failed"
        print("  -> PASSED: Password reset verified and authenticated.")

    # Restore researcher password back to nanstack
    import db
    rc, _ = db.generate_reset_token('researcher@nanstack.tech')
    db.reset_password_with_token('researcher@nanstack.tech', rc, 'nanstack')

    # ----------------------------------------------------
    # TEST 10: Logout Route
    # ----------------------------------------------------
    print("\n[TEST 10] GET /logout (Session invalidation)")
    resp = no_redirect_opener.open(f"{BASE_URL}/logout")
    assert resp.status == 302, f"Expected 302, got {resp.status}"
    print(f"  -> Logout Redirect: {resp.headers.get('Location')}")

    # Verify that trying to access /dashboard now fails
    resp_after = no_redirect_opener.open(f"{BASE_URL}/dashboard")
    assert resp_after.status == 302, "Dashboard should redirect after logout"
    print("  -> PASSED: Session cleared, user redirected to landing page.")

    print("\n" + "=" * 70)
    print("ALL 10 VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == '__main__':
    run_tests()
