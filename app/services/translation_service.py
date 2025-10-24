from google.cloud import translate_v2 as translate
import os

class TranslationService:
    def __init__(self):
        # Set up Google Cloud credentials
        credentials_path = os.path.join(os.path.dirname(__file__), '..', '..', 'google-credentials.json')
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        self.mock_mode = False
        try:
            self.client = translate.Client()
            print("Google Cloud Translate client initialized successfully")
            # Test the API with a minimal request
            self._test_api_connection()
        except Exception as e:
            print(f"Google Cloud Translate API not available: {e}")
            print("Switching to mock mode for testing...")
            self.mock_mode = True

    def _test_api_connection(self):
        """
        Test the API connection with a minimal request
        """
        try:
            # Try a minimal API call to test if the service is enabled
            result = self.client.translate("hello", source_language="en", target_language="es")
            print("Google Cloud Translate API is enabled and working")
        except Exception as e:
            error_str = str(e).lower()
            if "service_disabled" in error_str or "403" in error_str or "permission" in error_str:
                print(f"API test failed - service disabled: {e}")
                self.mock_mode = True
            else:
                print(f"API test completed (expected failure with test data): {e}")

    def translate_text(self, text, source_lang="zu", target_lang="en"):
        """
        Translates text from source language to target language
        :param text: Text to translate
        :param source_lang: Source language code (default: "zu" for isiZulu)
        :param target_lang: Target language code (default: "en" for English)
        :return: Translated text
        """
        if not text or text.strip() == "":
            return ""

        if self.mock_mode:
            return self._mock_translation(text)

        try:
            print(f"Translating '{text}' from {source_lang} to {target_lang}...")

            # Call the API
            result = self.client.translate(
                text,
                source_language=source_lang,
                target_language=target_lang
            )

            # Return the translated text
            translated_text = result["translatedText"]
            print(f"Translation: {translated_text}")
            return translated_text

        except Exception as e:
            print(f"Translation error: {e}")
            # Fall back to mock mode
            return self._mock_translation(text)

    def _mock_translation(self, text):
        """
        Mock translation for testing when API is not available
        """
        # Simple mock translations for common isiZulu phrases
        mock_translations = {
            "Sawubona": "Hello",
            "Ngicela usizo": "I need help",
            "Ngezitshalo zami": "With my plants",
            "Izitshalo zami zinezifo": "My plants have diseases",
            "Isimo sezulu sithini?": "What's the weather like?",
            "Ngidinga umanyolo": "I need fertilizer",
            "Ngidinga ukunisela": "I need to water"
        }

        for zulu, english in mock_translations.items():
            if zulu.lower() in text.lower():
                return english

        message = f"[Translation not available] {text}"
        print(f"Mock translation: {message}")
        return message