from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from getpass import getpass
from importlib import import_module
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

from kiteconnect import KiteConnect

LOGGER = logging.getLogger(__name__)
_DEFAULT_TIMEOUT_MS = 45_000


@dataclass(slots=True)
class KiteSession:
    """Authenticated Kite session used by candle APIs.

    Stores only non-sensitive session data. Credentials like password and PIN
    are not persisted. The ``credentials`` field stores only user_id, which is
    needed for session operations.
    """

    kite: KiteConnect
    session_cache: Dict[str, Any] = field(default_factory=dict)


class KiteSessionManager:
    """Manages Kite session creation with Playwright-based login automation.

    This class handles the complete login flow including credential validation,
    Playwright-based browser automation, and session initialization.
    """

    def __init__(self) -> None:
        """Initialize session manager and compose configuration.

        Configuration is composed internally from environment variables or
        user prompts if not set. Required config includes:
        - KITE_API_KEY: Kite API key
        - KITE_API_SECRET: Kite API secret
        - KITE_USER_ID: Zerodha user ID or email (for first login screen)
        - KITE_PASSWORD: Zerodha account password (optional if token cached)
        - KITE_PIN: 6-digit Kite Mobile App Code AND 2FA PIN (optional if token cached)

        Optional environment variables:
        - KITE_HEADLESS (bool, default False - shows browser window)
        - KITE_TIMEOUT_MS (int, default 45000)
        - KITE_REDIRECT_URL (str, default localhost:1130 with request_token)
        """
        self.session: KiteSession | None = None
        self.config = self._get_session_config()
        self.timeout_ms = int(self.config.get("timeout_ms", _DEFAULT_TIMEOUT_MS))
        self.headless = bool(self.config.get("headless", True))
        self.redirect_url = self.config.get("redirect_url", "http://localhost:1130/")
        self.browser_args = self.config.get(
            "browser_args", ["--disable-blink-features=AutomationControlled"]
        )
        # Access token artifact path
        self.artifact_dir = Path.home() / ".kite_session"
        self.artifact_dir.mkdir(exist_ok=True)
        self.token_file = self.artifact_dir / "access_token.json"

    @staticmethod
    def _get_env_or_prompt(
        name: str, prompt_text: str, secret: bool = False
    ) -> str | None:
        """Get configuration value from environment variable or user prompt.

        Args:
            name: Environment variable name.
            prompt_text: Text to display in prompt.
            secret: If True, hides input (for passwords/PINs).

        Returns:
            Configuration value from env or user input.
        """
        value = os.getenv(name)
        if value:
            return value

        if secret:

            return getpass(f"{prompt_text}: ")

        return input(f"{prompt_text}: ").strip() or None

    def _get_user_config(self) -> Dict[str, Any]:
        """Compose session configuration from environment or user input.

        Returns:
            Dict containing all configuration needed for session creation.
        """
        LOGGER.info(
            "Kite session env vars required for session creation: %s",
            ", ".join(
                [
                    "KITE_USER_ID",
                    "KITE_PASSWORD",
                    "KITE_PIN"
                ]
            ),
        )

        return {
            "user_id": self._get_env_or_prompt("KITE_USER_ID", "Kite user ID"),
            "password": self._get_env_or_prompt(
                "KITE_PASSWORD", "Kite password", secret=True
            ),
            "pin": self._get_env_or_prompt("KITE_PIN", "Kite PIN", secret=True),
        }

    def _get_session_config(self) -> Dict[str, Any]:
        """Compose session configuration from environment or user input.

        Returns:
            Dict containing all configuration needed for session creation.
        """
        LOGGER.info(
            "Kite session env vars required for session creation: %s",
            ", ".join(
                [
                    "KITE_API_KEY",
                    "KITE_API_SECRET",
                    "KITE_HEADLESS",
                    "KITE_TIMEOUT_MS",
                    "KITE_REDIRECT_URL",
                ]
            ),
        )

        return {
            "api_key": self._get_env_or_prompt("KITE_API_KEY", "Kite API key"),
            "api_secret": self._get_env_or_prompt(
                "KITE_API_SECRET", "Kite API secret", secret=True
            ),
            "headless": os.getenv("KITE_HEADLESS", "False").lower() == "true",
            "timeout_ms": int(os.getenv("KITE_TIMEOUT_MS", "45000")),
            "redirect_url": os.getenv(
                "KITE_REDIRECT_URL",
                "http://localhost:1130/",
            ),
        }

    def start_session(self, cli=True) -> KiteSession:
        """Create and return an authenticated Kite session.

        Tries to use cached access token if available and valid.
        If expired or missing, runs full login flow and caches new token.

        :param cli

        Returns:
            KiteSession: Authenticated session with access token and
                kite client.

        Raises:
            ValueError: If required config is missing.
            RuntimeError: If login fails or request token extraction fails.
        """
        if cli:
            return self._cli_login()
        return self._login()

    def _cli_login(self):
        kite = self._build_kite_client()

        # Try to load and validate cached access token
        print("\n" + "=" * 60)
        print("🔍 Checking for cached access token...")
        print("=" * 60)

        cached_token_data = self._load_access_token()
        if cached_token_data:
            access_token = cached_token_data.get("access_token")
            if access_token and self._validate_access_token(kite, access_token):
                print("✅ Cached access token is valid!")
                kite.set_access_token(access_token)
                print("=" * 60 + "\n")
                self.session = KiteSession(
                    kite=kite,
                    # session_cache=cached_token_data,
                )
                return self.session
            else:
                print("❌ Cached access token expired or invalid")

        # Run full login flow if no valid cached token
        # Validate that password and pin are provided for full login
        self.config = self.config | self._get_user_config()
        self._validate_config()

        print("\n" + "=" * 60)
        print("🔐 Running full login flow...")
        print("=" * 60)

        request_token = self._login_and_get_request_token(kite.login_url())

        print("\n" + "=" * 60)
        print("🔑 Exchanging request token for access token...")
        print("=" * 60)

        session_data = self._get_access_token(kite, request_token)
        access_token = str(session_data["access_token"])
        kite.set_access_token(access_token)

        # Save access token for future use
        self._save_access_token(session_data)

        print("✅ Access token obtained successfully!")
        print("=" * 60 + "\n")

        self.session = KiteSession(
            kite=kite,
            # session_cache=dict(session_data),
        )
        return self.session

    def _login(self):
        """Run login flow where user enters credentials manually in browser."""
        self.headless = False
        kite = self._build_kite_client()

        # Try to load and validate cached access token
        print("\n" + "=" * 60)
        print("🔍 Checking for cached access token...")
        print("=" * 60)

        cached_token_data = self._load_access_token()
        if cached_token_data:
            access_token = cached_token_data.get("access_token")
            if access_token and self._validate_access_token(kite, access_token):
                print("✅ Cached access token is valid!")
                kite.set_access_token(access_token)
                print("=" * 60 + "\n")
                self.session = KiteSession(
                    kite=kite,
                    # session_cache=cached_token_data,
                )
                return self.session
            else:
                print("❌ Cached access token expired or invalid")

        self._validate_api_config()

        print("\n" + "=" * 60)
        print("🔐 Running manual browser login flow...")
        print("=" * 60)

        request_token = self._manual_login_and_get_request_token(kite.login_url())

        print("\n" + "=" * 60)
        print("🔑 Exchanging request token for access token...")
        print("=" * 60)

        session_data = self._get_access_token(kite, request_token)
        access_token = str(session_data["access_token"])
        kite.set_access_token(access_token)

        # Save access token for future use
        self._save_access_token(session_data)

        print("✅ Access token obtained successfully!")
        print("=" * 60 + "\n")

        self.session = KiteSession(
            kite=kite,
            # session_cache=dict(session_data),
        )
        self.headless = self.config.get("headless", True)
        return self.session

    def _validate_api_config(self) -> None:
        """Validate config required for token exchange and API client setup."""
        required = ["api_key", "api_secret"]
        missing = [name for name in required if not self.config.get(name)]
        if missing:
            raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    def _validate_config(self) -> None:
        """Validate that all required config keys are present."""
        required = [
            "api_key",
            "api_secret",
            "user_id",
            "password",
            "pin",
        ]
        missing = [name for name in required if not self.config.get(name)]
        if missing:
            raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    def _build_kite_client(self) -> Any:
        """Create KiteConnect client via runtime import.

        Returns:
            KiteConnect: KiteConnect client instance.

        Raises:
            RuntimeError: If kiteconnect.KiteConnect is not available.
        """
        kite_module = import_module("kiteconnect")
        kite_class = getattr(kite_module, "KiteConnect", None)
        if kite_class is None:
            raise RuntimeError("kiteconnect.KiteConnect is not available")
        return kite_class(api_key=str(self.config["api_key"]))

    def _manual_login_and_get_request_token(self, kite_login_url: str) -> str:
        """Open Kite login page and let user complete auth manually."""
        if self.headless:
            raise RuntimeError(
                "Manual login requires a visible browser. Set KITE_HEADLESS=false."
            )

        playwright_sync = import_module("playwright.sync_api")
        sync_playwright = getattr(playwright_sync, "sync_playwright")

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=self.headless, args=list(self.browser_args)
            )
            page = browser.new_page()
            redirect_url = None

            def on_response(response):
                nonlocal redirect_url
                if response.status in [301, 302, 303, 307, 308]:
                    location = response.headers.get("location")
                    if location and "request_token" in location:
                        redirect_url = location

            page.on("response", on_response)
            try:
                page.goto(
                    kite_login_url,
                    wait_until="networkidle",
                    timeout=self.timeout_ms,
                )
                print(f"\n✓ Page loaded: {page.url}")
                print("ℹ️  Complete login manually in the opened browser window.")
                print("ℹ️  This includes user ID, password, and PIN/2FA.")

                deadline = datetime.now().timestamp() + (self.timeout_ms / 1000)
                while datetime.now().timestamp() < deadline:
                    current_url = page.url
                    if "request_token" in current_url:
                        redirect_url = current_url
                        break
                    if redirect_url:
                        break
                    page.wait_for_timeout(500)

                if not redirect_url:
                    current_url = page.url
                    if "request_token" in current_url:
                        redirect_url = current_url

                if not redirect_url:
                    raise RuntimeError(
                        "Timed out waiting for Kite request_token redirect. "
                        "Complete login in the browser and ensure redirect URL "
                        "is configured correctly."
                    )

                request_token = self._extract_request_token(redirect_url)
                if not request_token:
                    raise RuntimeError("Kite login redirect missing request_token")

                print("=" * 60)
                print("✅ Authentication Successful!")
                print("=" * 60 + "\n")
                return request_token
            finally:
                page.remove_listener("response", on_response)
                browser.close()

    def _login_and_get_request_token(self, kite_login_url: str) -> str:
        """Automate Kite login with Playwright and return request token.

        Args:
            kite_login_url: The Kite login URL from KiteConnect client.

        Returns:
            str: The request token extracted from the redirect URL.

        Raises:
            RuntimeError: If login times out or request token is missing.
        """
        playwright_sync = import_module("playwright.sync_api")
        sync_playwright = getattr(playwright_sync, "sync_playwright")
        playwright_timeout_error = getattr(
            playwright_sync,
            "TimeoutError",
            Exception,
        )

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=self.headless, args=list(self.browser_args)
            )
            page = browser.new_page()
            try:
                print("\n" + "=" * 60)
                print("🔐 Starting Kite Authentication...")
                print("=" * 60)
                print(f"Headless mode: {self.headless}")
                print(f"Timeout: {self.timeout_ms}ms")

                page.goto(
                    kite_login_url,
                    wait_until="networkidle",
                    timeout=self.timeout_ms,
                )
                print(f"\n✓ Page loaded: {page.url}")
                print("ℹ️  Watch the browser window for login events...\n")

                print("📝 Step 1: Filling User ID/Email...")
                self._fill_first(
                    page=page,
                    selectors=[
                        "input#userid:not([type='number'])",
                        "input[name='user_id']:not([type='number'])",
                        "input#user_id",
                        "input[type='text']:first-of-type",
                        "input[placeholder*='email']",
                        "input[placeholder*='user']",
                    ],
                    value=str(self.config["user_id"]),
                    field_name="User ID/Email",
                )

                print("🔘 Step 2: Clicking Next button...")
                self._click_first(
                    page=page,
                    selectors=[
                        "button[type='submit']",
                        "button:has-text('Next')",
                        "button:has-text('Continue')",
                        "button:has-text('Login')",
                        "input[value='Login']",
                        "button[class*='submit']",
                    ],
                    button_name="Next",
                )

                # Wait for password field to appear (it may be on a new screen)
                print("⏳ Waiting for password field to appear...")
                try:
                    page.wait_for_selector(
                        "input[type='password']", timeout=self.timeout_ms
                    )
                except:
                    pass  # Try filling anyway if wait fails

                page.wait_for_timeout(500)  # Small delay for screen transition
                print("📝 Step 3: Filling Password...")
                self._fill_first(
                    page=page,
                    selectors=[
                        "input#password",
                        "input[name='password']",
                        "input[type='password']",
                        'input[placeholder*="Password"]',
                    ],
                    value=str(self.config["password"]),
                    field_name="Password",
                )

                print("🔘 Step 4: Clicking Next button...")
                self._click_first(
                    page=page,
                    selectors=[
                        "button[type='submit']",
                        "button:has-text('Next')",
                        "button:has-text('Continue')",
                        "button:has-text('Login')",
                        "input[value='Login']",
                        "button[class*='submit']",
                    ],
                    button_name="Next",
                )

                # Wait for Mobile App Code field to appear (third screen)
                print("⏳ Waiting for Mobile App Code field (6-digit) to appear...")
                try:
                    page.wait_for_selector(
                        "input#userid[type='number']", timeout=self.timeout_ms
                    )
                except:
                    pass  # Try filling anyway if wait fails

                page.wait_for_timeout(500)  # Small delay for screen transition
                print("📝 Step 5: Filling 6-digit Mobile App Code (PIN)...")
                self._fill_first(
                    page=page,
                    selectors=[
                        "input#userid[type='number']",
                        "input[label='Mobile App Code']",
                        "input[type='number']",
                        'input[placeholder*="••••••"]',
                    ],
                    value=str(self.config["pin"]),
                    field_name="Mobile App Code",
                )

                print("⏳ Step 6: Waiting for automatic redirect...")
                redirect_url = None

                def on_response(response):
                    nonlocal redirect_url
                    # Capture redirect responses
                    if response.status in [301, 302, 303, 307, 308]:
                        location = response.headers.get("location")
                        if location and "request_token" in location:
                            redirect_url = location

                page.on("response", on_response)

                try:
                    # Wait for navigation (which may fail due to connection refused)
                    page.wait_for_timeout(self.timeout_ms)

                    # If we captured the redirect, use it
                    if redirect_url:
                        print(f"✓ Captured redirect: {redirect_url[:80]}...\n")
                    else:
                        # Try to get current URL as fallback
                        current_url = page.url
                        if "request_token" in current_url:
                            redirect_url = current_url
                            print(
                                f"✓ Current URL contains token: {current_url[:80]}...\n"
                            )
                        else:
                            raise RuntimeError(
                                "Could not capture Kite redirect URL. "
                                "Check credentials and API configuration."
                            )
                except Exception as e:
                    # If navigation fails but we have the redirect, use it
                    if not redirect_url:
                        raise RuntimeError(
                            "Timed out waiting for Kite request_token redirect. "
                            "Check credentials, PIN/TOTP requirements, "
                            "and API app configuration."
                        ) from e
                    print(
                        f"✓ Captured redirect despite error: {redirect_url[:80]}...\n"
                    )
                finally:
                    page.remove_listener("response", on_response)

                request_token = self._extract_request_token(redirect_url)
                if not request_token:
                    raise RuntimeError("Kite login redirect missing request_token")

                print("=" * 60)
                print("✅ Authentication Successful!")
                print("=" * 60 + "\n")

                return request_token
            finally:
                browser.close()

    def _extract_request_token(self, url: str) -> str | None:
        """Extract request token from redirect URL.

        Args:
            url: The Kite redirect URL.

        Returns:
            str | None: The request token if present, None otherwise.
        """
        query = parse_qs(urlparse(url).query)
        tokens = query.get("request_token")
        if not tokens:
            return None
        return tokens[0]

    def _get_access_token(self, kite: Any, request_token: str) -> dict:
        """Exchange request token for access token using KiteConnect.

        Args:
            kite: KiteConnect client instance.
            request_token: The request token from redirect URL.

        Returns:
            dict: Session data including access_token and public_token.

        Raises:
            RuntimeError: If session generation fails.
        """
        try:
            session_data = kite.generate_session(
                request_token=request_token,
                api_secret=str(self.config["api_secret"]),
            )
            return session_data
        except Exception as e:
            raise RuntimeError(
                f"Failed to exchange request token for access token: {e}"
            )

    def _save_access_token(self, session_data: dict) -> None:
        """Save access token and related data to local artifact.

        Args:
            session_data: Session data from kite.generate_session().
        """
        try:
            token_data = {
                "access_token": session_data.get("access_token"),
                "timestamp": datetime.utcnow().isoformat(),
            }
            with open(self.token_file, "w") as f:
                json.dump(token_data, f)
            print(f"💾 Access token saved to {self.token_file}")
        except Exception as e:
            print(f"⚠️  Failed to save access token: {e}")

    def _load_access_token(self) -> dict | None:
        """Load cached access token from local artifact.

        Returns:
            dict | None: Token data if exists, None otherwise.
        """
        if not self.token_file.exists():
            print("ℹ️  No cached access token found")
            return None

        try:
            with open(self.token_file, "r") as f:
                token_data = json.load(f)
            print("📖 Loaded cached access token from disk")
            return token_data
        except Exception as e:
            print(f"⚠️  Failed to load cached access token: {e}")
            return None

    def _validate_access_token(self, kite: Any, access_token: str) -> bool:
        """Validate access token by making a test API call.

        Args:
            kite: KiteConnect client instance.
            access_token: The access token to validate.

        Returns:
            bool: True if token is valid, False otherwise.
        """
        try:
            kite.set_access_token(access_token)
            # Test API call to validate token
            profile = kite.profile()
            if profile and profile.get("user_id"):
                print("✓ Access token validation successful")
                return True
            return False
        except Exception as e:
            print(f"✗ Access token validation failed: {e}")
            return False

    def _fill_first(
        self, page: Any, selectors: list[str], value: str, field_name: str = ""
    ) -> None:
        """Fill the first matching input field with a value.

        Args:
            page: Playwright page object.
            selectors: List of CSS selectors to try.
            value: Value to fill in the input.
            field_name: Human-readable name of the field for logging.

        Raises:
            RuntimeError: If no matching input is found.
        """
        for selector in selectors:
            try:
                locator = page.locator(selector)
                if locator.count() > 0:
                    locator.first.fill(value)
                    if field_name:
                        print(f"✓ Filled {field_name} using selector: {selector}")
                    return
            except Exception as e:
                continue

        # Debug output: Show available input elements on the page
        print(f"\n❌ Unable to find {field_name} input on the page.")
        print("\nAvailable input elements:")
        try:
            all_inputs = page.locator("input").all()
            for i, input_elem in enumerate(all_inputs, 1):
                input_type = input_elem.get_attribute("type") or "text"
                input_name = input_elem.get_attribute("name") or "(no name)"
                input_id = input_elem.get_attribute("id") or "(no id)"
                input_placeholder = (
                    input_elem.get_attribute("placeholder") or "(no placeholder)"
                )
                print(
                    f"  {i}. type='{input_type}' name='{input_name}' id='{input_id}' placeholder='{input_placeholder}'"
                )
        except Exception as e:
            print(f"  (Could not inspect inputs: {e})")

        error_msg = f"Unable to find input with selectors: {selectors}"
        if field_name:
            error_msg = f"Unable to find {field_name} input with selectors: {selectors}"
        raise RuntimeError(error_msg)

    def _click_first(
        self, page: Any, selectors: list[str], button_name: str = ""
    ) -> None:
        """Click the first matching button element.

        Args:
            page: Playwright page object.
            selectors: List of CSS selectors to try.
            button_name: Human-readable name of the button for logging.

        Raises:
            RuntimeError: If no matching button is found.
        """
        for selector in selectors:
            try:
                locator = page.locator(selector)
                if locator.count() > 0:
                    locator.first.click()
                    if button_name:
                        print(
                            f"✓ Clicked {button_name} button using selector: {selector}"
                        )
                    return
            except Exception as e:
                continue

        # Debug output: Show available button elements on the page
        print(f"\n❌ Unable to find {button_name} button on the page.")
        print("\nAvailable buttons:")
        try:
            all_buttons = page.locator(
                "button, input[type='submit'], input[type='button']"
            ).all()
            for i, btn in enumerate(all_buttons, 1):
                btn_type = btn.get_attribute("type") or "button"
                btn_text = btn.text_content() or ""
                btn_id = btn.get_attribute("id") or "(no id)"
                btn_class = btn.get_attribute("class") or "(no class)"
                btn_value = btn.get_attribute("value") or ""
                display_text = btn_text.strip() or btn_value or "(no text)"
                print(
                    f"  {i}. type='{btn_type}' id='{btn_id}' text='{display_text}' class='{btn_class}'"
                )
        except Exception as e:
            print(f"  (Could not inspect buttons: {e})")

        error_msg = f"Unable to find button with selectors: {selectors}"
        if button_name:
            error_msg = (
                f"Unable to find {button_name} button with selectors: {selectors}"
            )
        raise RuntimeError(error_msg)
        raise RuntimeError(error_msg)
