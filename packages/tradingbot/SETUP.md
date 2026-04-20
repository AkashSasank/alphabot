# Setup Guide for tradingbot with uv

## Quick Start (Recommended)

Run the setup script that handles both installation and initialization:

```bash
chmod +x install.sh
./install.sh
```

This will:
1. Install the package with uv
2. Run initialization to install Playwright browsers
3. Display next steps

## Manual Installation

If you prefer to install manually:

### Step 1: Install the package

```bash
uv pip install -e .
```

### Step 2: Run initialization

```bash
uv run tradingbot-init
```

Or using Python module directly:
```bash
uv run python -m tradingbot.cli
```

This installs Playwright Chromium browser (required for Kite login automation).

## Configuration

### Step 1: Create .env file from template

```bash
cp .env.example .env
```

### Step 2: Get your credentials

**Kite API Credentials:**
1. Go to https://kite.zerodha.com/settings/developer/applications
2. Copy your API Key and API Secret

**Zerodha User ID/Email:**
- This is your login username or email address (what you use to login to Kite on the web)

**6-digit Kite PIN:**
1. Open the Kite mobile app on your phone
2. Go to **Settings** → **Profile** → **Your Profile**
3. Find **"Mobile App Code"** (exactly 6 digits)
4. **Note:** This same 6-digit code is used for BOTH:
   - The **Mobile App Code** field (third screen)
   - The **2FA PIN** field (fourth screen)

### Step 3: Update .env with your credentials

```
KITE_API_KEY=your_api_key_here
KITE_API_SECRET=your_api_secret_here
KITE_USER_ID=your_username_or_email@example.com
KITE_PIN=123456
KITE_HEADLESS=False
KITE_TIMEOUT_MS=45000

# Optional: Configure the redirect URL after successful authentication
# KITE_REDIRECT_URL=http://localhost:1130/?action=login&type=login&status=success&request_token=*
```

### Step 4: Run the application

```bash
uv run python main.py
```

**You will be prompted for:**
- **Zerodha Password** (hidden input)
- **6-digit Kite PIN** (visible input - used for both Mobile App Code and 2FA)

**Login flow in browser (watch the window):**
1. First screen: User ID/Email (from KITE_USER_ID env var)
2. Second screen: Password (you'll enter this)
3. Third screen: 6-digit Mobile App Code (from KITE_PIN env var)
4. Fourth screen: 2FA PIN (you'll enter this)

## Access Token Caching

After successful login, the access token is automatically saved locally to speed up future sessions. When you run the app again:

- ✅ **If cached token is valid**: Skips the full login flow and reuses the cached access token
- ❌ **If cached token is expired**: Automatically runs the full login flow again and saves the new token

**Cached token location:**
```
~/.kite_session/access_token.json
```

This means subsequent runs (same day) will be instant - no browser window, no manual login!

## Building the package

Build using uv:
```bash
uv build
```

This uses hatchling as the build backend configured in pyproject.toml.

## Troubleshooting

If the init command fails:

1. **Ensure Playwright is installed:**
   ```bash
   uv run python -m playwright install chromium
   ```

2. **Check Python version (requires 3.10+):**
   ```bash
   python --version
   ```

3. **Rebuild the package:**
   ```bash
   uv pip install --force-reinstall -e .
   ```



