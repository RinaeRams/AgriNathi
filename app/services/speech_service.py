from google.cloud import speech_v1p1beta1 as speech
import os

class SpeechToTextService:
    def __init__(self):
        # Initialize the Google Cloud Speech client
        self.mock_mode = False
        try:
            # Set credentials path
            credentials_path = os.path.join(os.path.dirname(__file__), '..', '..', 'google-credentials.json')
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            self.client = speech.SpeechClient()
            print("Google Cloud Speech client initialized successfully")
            # Test the API with a minimal request to check if it's enabled
            self._test_api_connection()
        except Exception as e:
            print(f"Google Cloud Speech API not available: {e}")
            print("Switching to mock mode for testing...")
            self.mock_mode = True

    def _test_api_connection(self):
        """
        Test the API connection with a minimal request
        """
        try:
            # Try a minimal API call to test if the service is enabled
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code='en-US',  # Use English for testing
            )
            audio = speech.RecognitionAudio(content=b'test')  # Minimal content

            # This will fail if API is disabled, triggering our error handling
            self.client.recognize(config=config, audio=audio)
            print("Google Cloud Speech API is enabled and working")
        except Exception as e:
            error_str = str(e).lower()
            if "service_disabled" in error_str or "403" in error_str:
                print(f"API test failed - service disabled: {e}")
                self.mock_mode = True
            else:
                print(f"API test completed (expected failure with test data): {e}")

    def transcribe_audio(self, audio_file_path, language_code='zu-ZA'):
        """
        Transcribes audio file to text using Google Cloud Speech API
        :param audio_file_path: Path to the audio file
        :param language_code: Language code for isiZulu (zu-ZA)
        :return: Transcribed text
        """
        print(f"DEBUG: Starting transcription for file: {audio_file_path}")
        print(f"DEBUG: Mock mode: {self.mock_mode}")
        print(f"DEBUG: Requested language: {language_code}")

        if self.mock_mode:
            return self._mock_transcription(audio_file_path)

        try:
            # Read the audio file
            with open(audio_file_path, 'rb') as audio_file:
                content = audio_file.read()

            print(f"DEBUG: Audio file size: {len(content)} bytes")
            print(f"DEBUG: First 10 bytes (hex): {content[:10].hex() if len(content) >= 10 else 'N/A'}")

            # Determine encoding based on file content
            if content.startswith(b'\x1a\x45\xdf\xa3'):
                print("DEBUG: Detected WebM format")
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
                    sample_rate_hertz=48000,
                    language_code=language_code,
                    enable_automatic_punctuation=True,
                    enable_word_time_offsets=False,
                )
            elif content.startswith(b'RIFF'):
                print("DEBUG: Detected WAV format")
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    language_code=language_code,
                    enable_automatic_punctuation=True,
                    enable_word_time_offsets=False,
                )
            elif content.startswith(b'\x00\x00\x00'):
                print("DEBUG: Detected MP4 format")
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.MP3,
                    sample_rate_hertz=16000,
                    language_code=language_code,
                    enable_automatic_punctuation=True,
                    enable_word_time_offsets=False,
                )
            else:
                print("DEBUG: Unknown format, defaulting to WebM_OPUS")
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
                    sample_rate_hertz=48000,
                    language_code=language_code,
                    enable_automatic_punctuation=True,
                    enable_word_time_offsets=False,
                )

            # Configure the audio settings
            audio = speech.RecognitionAudio(content=content)

            # Try English first for better recognition, then isiZulu if needed
            print(f"DEBUG: Trying English recognition first for better accuracy")

            # Create English config
            english_config = speech.RecognitionConfig(
                encoding=config.encoding,
                sample_rate_hertz=config.sample_rate_hertz,
                language_code='en-US',
                enable_automatic_punctuation=True,
                enable_word_time_offsets=False,
            )

            print(f"DEBUG: Sending English request to Google Cloud Speech API with config: encoding={english_config.encoding}, sample_rate={getattr(english_config, 'sample_rate_hertz', 'unspecified')}, language={english_config.language_code}")
            english_response = self.client.recognize(config=english_config, audio=audio)
            print(f"DEBUG: English API response received: {len(english_response.results)} results")

            # Extract English transcript
            english_transcript = ""
            english_confidence = 0.0
            for i, result in enumerate(english_response.results):
                print(f"DEBUG: English result {i}: confidence={result.alternatives[0].confidence if result.alternatives else 'N/A'}, transcript='{result.alternatives[0].transcript if result.alternatives else 'N/A'}'")
                if result.alternatives:
                    english_transcript += result.alternatives[0].transcript
                    english_confidence = max(english_confidence, result.alternatives[0].confidence)

            english_transcript = english_transcript.strip()
            print(f"DEBUG: English transcript: '{english_transcript}' (confidence: {english_confidence})")

            # Now try isiZulu
            print(f"DEBUG: Sending isiZulu request to Google Cloud Speech API with config: encoding={config.encoding}, sample_rate={getattr(config, 'sample_rate_hertz', 'unspecified')}, language={config.language_code}")
            response = self.client.recognize(config=config, audio=audio)
            print(f"DEBUG: isiZulu API response received: {len(response.results)} results")

            # Extract isiZulu transcript
            zulu_transcript = ""
            zulu_confidence = 0.0
            for i, result in enumerate(response.results):
                print(f"DEBUG: isiZulu result {i}: confidence={result.alternatives[0].confidence if result.alternatives else 'N/A'}, transcript='{result.alternatives[0].transcript if result.alternatives else 'N/A'}'")
                if result.alternatives:
                    zulu_transcript += result.alternatives[0].transcript
                    zulu_confidence = max(zulu_confidence, result.alternatives[0].confidence)

            zulu_transcript = zulu_transcript.strip()
            print(f"DEBUG: isiZulu transcript: '{zulu_transcript}' (confidence: {zulu_confidence})")

            # Choose the better result based on confidence and content
            if english_confidence > zulu_confidence and english_transcript:
                print(f"DEBUG: Using English result (higher confidence: {english_confidence} vs {zulu_confidence})")
                final_transcript = english_transcript
            elif zulu_transcript:
                print(f"DEBUG: Using isiZulu result (confidence: {zulu_confidence})")
                final_transcript = zulu_transcript
            elif english_transcript:
                print(f"DEBUG: Using English result (only available option)")
                final_transcript = english_transcript
            else:
                print(f"DEBUG: No valid transcripts found")
                final_transcript = ""

            print(f"DEBUG: Final transcript: '{final_transcript}'")

            # If no transcript and file is small, it might be too short
            if not final_transcript and len(content) < 50000:  # Less than 50KB
                print(f"DEBUG: Audio file too small ({len(content)} bytes), likely no speech captured")
                return ""

            return final_transcript

        except Exception as e:
            error_str = str(e).lower()
            # Check if this is an API disabled or authentication error
            if "service_disabled" in error_str or "403" in error_str or "not enabled" in error_str:
                print(f"Google Cloud API error detected: {e}")
                print("Switching to mock mode for this request...")
                self.mock_mode = True
                return self._mock_transcription(audio_file_path)
            elif "sample_rate_hertz" in error_str and "must either be unspecified" in error_str:
                # Handle sample rate mismatch - try without specifying sample rate
                print("Sample rate mismatch detected for file, trying without sample_rate_hertz...")
                try:
                    config_no_rate = speech.RecognitionConfig(
                        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                        language_code=language_code,
                        enable_automatic_punctuation=True,
                        enable_word_time_offsets=False,
                    )
                    response = self.client.recognize(config=config_no_rate, audio=audio)
                    print(f"Retry API response received: {len(response.results)} results")

                    transcript = ""
                    for result in response.results:
                        print(f"Retry result confidence: {result.alternatives[0].confidence if result.alternatives else 'N/A'}")
                        transcript += result.alternatives[0].transcript

                    print(f"Retry final transcript: '{transcript.strip()}'")
                    final_transcript = transcript.strip()
                    if not final_transcript:  # If still empty, use mock
                        print("API returned empty transcript, using mock response...")
                        return self._mock_transcription(audio_file_path)
                    return final_transcript
                except Exception as e2:
                    print(f"Retry also failed: {e2}")
                    raise Exception(f"Error transcribing audio: {str(e)}")
            else:
                raise Exception(f"Error transcribing audio: {str(e)}")

    def transcribe_audio_stream(self, audio_stream, language_code='zu-ZA'):
        """
        Transcribes audio stream to text
        :param audio_stream: Audio stream data
        :param language_code: Language code
        :return: Transcribed text
        """
        print(f"DEBUG: Starting stream transcription, data size: {len(audio_stream)} bytes")
        print(f"DEBUG: First 10 bytes (hex): {audio_stream[:10].hex() if len(audio_stream) >= 10 else 'N/A'}")
        print(f"DEBUG: Mock mode: {self.mock_mode}")

        if self.mock_mode:
            return self._mock_transcription("stream")

        try:
            audio = speech.RecognitionAudio(content=audio_stream)

            # For browser-recorded audio streams, it's almost always WebM/Opus
            # Force WebM_OPUS encoding for all stream data from browser
            print("DEBUG: Using WEBM_OPUS encoding for browser stream audio")
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
                sample_rate_hertz=48000,  # Standard for WebM/Opus
                language_code=language_code,
                enable_automatic_punctuation=True,
            )

            print(f"DEBUG: Sending stream request to Google Cloud Speech API with config: encoding={config.encoding}, sample_rate={getattr(config, 'sample_rate_hertz', 'unspecified')}, language={config.language_code}")
            response = self.client.recognize(config=config, audio=audio)
            print(f"DEBUG: Stream API response received: {len(response.results)} results")

            transcript = ""
            for i, result in enumerate(response.results):
                print(f"Stream result {i}: confidence={result.alternatives[0].confidence if result.alternatives else 'N/A'}, transcript='{result.alternatives[0].transcript if result.alternatives else 'N/A'}'")
                if result.alternatives:
                    transcript += result.alternatives[0].transcript

            print(f"DEBUG: Stream final transcript: '{transcript.strip()}'")
            return transcript.strip()

        except Exception as e:
            error_str = str(e).lower()
            # Check if this is an API disabled or authentication error
            if "service_disabled" in error_str or "403" in error_str or "not enabled" in error_str:
                print(f"Google Cloud API error detected: {e}")
                print("Switching to mock mode for this request...")
                self.mock_mode = True
                return self._mock_transcription("stream")
            elif "sample_rate_hertz" in error_str and "must either be unspecified" in error_str:
                # Handle sample rate mismatch for WEBM OPUS - try without specifying sample rate
                print("Sample rate mismatch detected, trying without sample_rate_hertz...")
                try:
                    config_no_rate = speech.RecognitionConfig(
                        encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
                        language_code=language_code,
                        enable_automatic_punctuation=True,
                    )
                    response = self.client.recognize(config=config_no_rate, audio=audio)
                    print(f"Stream retry API response received: {len(response.results)} results")

                    transcript = ""
                    for i, result in enumerate(response.results):
                        print(f"Stream retry result {i}: confidence={result.alternatives[0].confidence if result.alternatives else 'N/A'}, transcript='{result.alternatives[0].transcript if result.alternatives else 'N/A'}'")
                        if result.alternatives:
                            transcript += result.alternatives[0].transcript

                    print(f"Stream retry final transcript: '{transcript.strip()}'")
                    final_transcript = transcript.strip()
                    if not final_transcript:  # If still empty, use mock
                        print("API returned empty transcript, using mock response...")
                        return self._mock_transcription("stream")
                    return final_transcript
                except Exception as e2:
                    print(f"Retry also failed: {e2}")
                    raise Exception(f"Error transcribing audio stream: {str(e)}")
            else:
                raise Exception(f"Error transcribing audio stream: {str(e)}")

    def _mock_transcription(self, source):
        """
        Mock transcription for testing when API is not available
        Returns a message indicating API is not available
        """
        message = "Speech-to-Text API is not available. Please enable Google Cloud Speech-to-Text API."
        print(f"Mock transcription for {source}: {message}")
        return message