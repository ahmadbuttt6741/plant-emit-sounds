# PhytoEcho AI: Bio-Acoustic Plant Stress Diagnostic System
**Developed by [Nanstack](https://nanstack.tech)**

An enterprise AI-driven bio-acoustic monitoring platform that listens to plant stem micro-vibrations and acoustic cavitation clicks using a 27mm piezoelectric contact transducer to detect, classify, and diagnose botanical distress before visible wilting occurs.

Based on the research and engineering blueprint: `AI based Bio-Acoustic sensor in plants.docx`.

---

## 🌟 Key Platform Features

1. **Enterprise SaaS Landing Page & Authentication**:
   - Modern 2-column SaaS layout introducing the bio-acoustic sensing methodology, 22-dimensional DSP extraction, and research highlights.
   - Secure login portal with password hashing (`werkzeug.security`), persistent SQLite user storage, and 30-day session cookies ("Remember Me").
   - 6-digit verification code password reset workflow.
   - 1-click test credentials auto-fill and direct demo sign-in for seamless evaluation.
2. **Strict Route & API Protection**:
   - `/` serves the public Landing Page & Login Portal for unauthenticated visitors.
   - Authenticated sessions auto-redirect directly to `/dashboard`.
   - Direct access to `/dashboard`, `/uploads/*`, and all `/api/*` telemetry endpoints requires active authentication (unauthorized requests return HTTP 401 or redirect to `/`).
3. **Multi-Phenotype Botanical Stress Classification**:
   - Real-time classification across 5 biological conditions:
     - 🌱 **Healthy Baseline**: Fluid micro-oscillations, regular vascular sap transport.
     - 💧 **Drought Stress**: Discrete xylem cavitation clicks (*"tik-tik"*) under water tension.
     - ☀️ **Heat Stress**: Rapid burst cavitation followed by thermal defense quietness.
     - 🐛 **Pest Attack**: Continuous mechanical scraping and chewing vibration (*"krrr"*).
     - ⚡ **Physical Impact / Attack**: High-amplitude structural shockwaves (*"DHAM spike"*).
4. **22-Dimensional Audio Feature Extraction Engine**:
   - High-pass filtering above 1.5 kHz for impulse isolation.
   - Cavitation click peak detection, click rate, and Inter-Click Interval (ICI) variance.
   - Spectral Centroid, Crest Factor, Skewness, Kurtosis, Zero-Crossing Rate (ZCR).
   - 5-band Mel frequency sub-band energy distribution.
5. **Interactive Telemetry Dashboard**:
   - HTML5 Canvas audio oscillogram with microsecond cavitation click pin markers.
   - 2D Mel-spectrogram spectral density heatmap.
   - 28-band FFT frequency spectrum visualizer.
   - Actionable agronomic prescriptions and physiological mitigation steps.
   - Prominent Nanstack branding and link to [nanstack.tech](https://nanstack.tech).

---

## 🔐 Default Demo Accounts

The system comes pre-configured with user accounts stored in SQLite (`demo/users.db`):

| Role | Username / Email | Password | Access Level |
| :--- | :--- | :--- | :--- |
| **Botany Administrator** | `botany` *(or `botany@nanstack.tech`)* | **`nanstack`** | Full Administrator Access |
| **Nanstack Administrator** | `admin` *(or `admin@nanstack.tech`)* | **`nanstack`** | Bio-Acoustic Lead |
| **Plant Physiologist** | `researcher` *(or `researcher@nanstack.tech`)* | **`nanstack`** | Research & Diagnostics |

> *Tip: On the landing page, the login inputs are pre-filled with **`botany`** / **`nanstack`**, or you can click **"Instant 1-Click Demo Login"** for immediate access.*

---

## 🚀 How to Run the Application

### Option 1: One-Click Startup (Windows)
Double-click `start_demo.bat` in the project root directory.

### Option 2: Command Line
```powershell
# Navigate to the workspace
cd "d:\UON\botany2\Projects\Plants emit sounds"

# Launch the Flask application
python demo/app.py
```

Then visit:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🧪 Automated Verification Suite

To run the complete automated test suite covering all 10 security, route-guard, and authentication checks:

```powershell
python demo/verify_auth.py
```

**Verified Checks:**
1. Public Landing Page renders with 200 OK and Nanstack branding.
2. Unauthenticated direct access to `/dashboard` is blocked and redirected (302).
3. Unauthenticated API calls (`/api/model-metrics`, `/api/analyze-sample/...`) return HTTP 401 Unauthorized.
4. Invalid credentials rejected with appropriate error alerts.
5. Valid credentials successfully authenticate and generate encrypted session cookies.
6. Authenticated sessions grant full access to `/dashboard` with user avatar & role.
7. Protected machine learning APIs execute accurately with session authentication.
8. Authenticated users visiting `/` are automatically forwarded to `/dashboard`.
9. Password reset workflow generates verification codes and updates the database.
10. Logout clears the session and returns to the landing page.

---

## 🌐 Route Architecture

- `GET /` — Public Landing Page & Login Portal (auto-redirects to `/dashboard` if signed in).
- `GET/POST /login` — Standard login route.
- `POST /api/auth/login` — JSON endpoint for async AJAX authentication.
- `POST /api/auth/forgot-password` — Generates a 6-digit password verification code.
- `POST /api/auth/reset-password` — Validates code and updates password in SQLite.
- `GET /logout` — Invalidation endpoint; clears session and redirects to `/?logged_out=1`.
- `GET /dashboard` — **[PROTECTED]** Full PhytoEcho AI Bio-Acoustic console.
- `GET /api/model-metrics` — **[PROTECTED]** Model evaluation metrics & 5-fold CV statistics.
- `GET /api/analyze-sample/<filename>` — **[PROTECTED]** Runs DSP & inference on preset canonical WAV files.
- `POST /api/upload` — **[PROTECTED]** Handles custom WAV/MP3 uploads or browser mic captures.
- `GET /uploads/<filename>` — **[PROTECTED]** Secure file delivery for authenticated sessions.

---

## 🔬 Hardware Specifications (ESP32 & Transducer)

When deploying to physical hardware in the greenhouse or field:
1. **Transducer**: 27mm Brass Piezoelectric disc clamped to semi-woody stems using micropore medical tape and acoustic coupling gel.
2. **Signal Conditioning**:
   - $1\text{ M}\Omega$ parallel discharge resistor across the piezo terminals.
   - $1.65\text{ V}$ DC virtual ground bias network (two $100\text{ k}\Omega$ voltage dividers + $10\ \mu\text{F}$ bypass capacitor) to center AC audio waveforms within the ESP32 ADC $0 - 3.3\text{ V}$ safe range.
3. **Edge Inference**:
   - ESP32 ADC timer-driven DMA sampling at 8 kHz – 22.05 kHz.
# plant-emit-sounds
