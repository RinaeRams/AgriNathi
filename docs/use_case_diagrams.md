# AgriNathi System - Use Case Diagrams

## Overview
AgriNathi is an agricultural voice assistant system that provides farmers with voice-based agricultural advice, weather information, plant disease detection, and market data. The system consists of multiple modules working together to serve farmers in rural areas.

## System Modules

### 1. Voice Recognition Module

```mermaid
graph TD
    A[Farmer] --> B[Voice Recognition System]

    B --> C[Record Audio]
    B --> D[Process Audio]
    B --> E[Transcribe to Text]
    B --> F[Translate to English]
    B --> G[Generate Agricultural Advice]

    C --> H[Web Browser/Microphone]
    D --> I[Google Speech-to-Text API]
    E --> I
    F --> J[Google Translate API]
    G --> K[Advice Engine]

    K --> L[Return Response]
    L --> A

    M[Extension Officer] --> N[Monitor Voice Queries]
    N --> O[Access Analytics]
    O --> P[Database]
```

**Use Cases:**
- **UC-1:** Record Voice Query
- **UC-2:** Transcribe isiZulu Speech
- **UC-3:** Translate to English
- **UC-4:** Generate Agricultural Advice
- **UC-5:** Monitor System Usage

### 2. Weather Information Module

```mermaid
graph TD
    A[Farmer] --> B[Weather System]

    B --> C[Select Location]
    B --> D[Fetch Weather Data]
    B --> E[Display Current Weather]
    B --> F[Display 5-Day Forecast]
    B --> G[Generate Farming Advice]

    C --> H[Location Input/Search]
    D --> I[OpenWeatherMap API]
    E --> J[Weather Display]
    F --> J
    G --> K[Advice Engine]

    K --> L[Display Recommendations]
    L --> A

    M[Weather Data] --> N[Cache System]
    N --> O[Local Storage]
```

**Use Cases:**
- **UC-6:** Search Weather by Location
- **UC-7:** View Current Weather
- **UC-8:** View Weather Forecast
- **UC-9:** Get Weather-based Farming Advice
- **UC-10:** Cache Weather Data

### 3. Plant Disease Scanner Module

```mermaid
graph TD
    A[Farmer] --> B[Plant Scanner System]

    B --> C[Capture Plant Image]
    B --> D[Upload Image]
    B --> E[Analyze Image]
    B --> F[Identify Disease]
    B --> G[Generate Treatment Plan]

    C --> H[Camera/Mobile Device]
    D --> I[File Upload Interface]
    E --> J[Image Processing Engine]
    F --> K[Disease Database]
    G --> L[Treatment Database]

    J --> M[AI/ML Model]
    K --> N[Plant Disease Library]
    L --> O[Treatment Guidelines]

    P[Results] --> Q[Display Diagnosis]
    Q --> R[Show Treatment Steps]
    R --> S[Provide Prevention Tips]
    S --> A

    T[Agricultural Expert] --> U[Update Disease Database]
    U --> N
    V[Update Treatment Database]
    V --> O
```

**Use Cases:**
- **UC-11:** Capture Plant Image
- **UC-12:** Upload Plant Image
- **UC-13:** Analyze Plant Disease
- **UC-14:** View Diagnosis Results
- **UC-15:** Get Treatment Recommendations
- **UC-16:** Update Disease Database

### 4. Mobile Application Module

```mermaid
graph TD
    A[Farmer] --> B[Mobile App]

    B --> C[Home Screen]
    B --> D[Voice Assistant]
    B --> E[Camera Screen]
    B --> F[Weather Screen]
    B --> G[Market Screen]

    C --> H[Navigation Menu]
    D --> I[Voice Recording]
    E --> J[Plant Scanning]
    F --> K[Weather Information]
    G --> L[Market Prices]

    I --> M[Send to Backend]
    J --> N[Send Image to Backend]
    K --> O[Fetch Weather Data]
    L --> P[Fetch Market Data]

    M --> Q[Voice Recognition Service]
    N --> R[Plant Analysis Service]
    O --> S[Weather Service]
    P --> T[Market Data Service]

    U[Response] --> V[Display Results]
    V --> A

    W[Offline Mode] --> X[Cache Data]
    X --> Y[Local Storage]
```

**Use Cases:**
- **UC-17:** Navigate App Sections
- **UC-18:** Use Voice Assistant
- **UC-19:** Scan Plants with Camera
- **UC-20:** Check Weather Information
- **UC-21:** View Market Prices
- **UC-22:** Work Offline

### 5. IVR (Interactive Voice Response) Module

```mermaid
graph TD
    A[Farmer] --> B[IVR System]

    B --> C[Call AgriNathi]
    B --> D[Language Selection]
    B --> E[Menu Navigation]
    B --> F[Voice Query]
    B --> G[Get Information]

    C --> H[Phone Call]
    D --> I[isiZulu/English]
    E --> J[Menu Options]
    F --> K[Voice Recording]
    G --> L[Information Retrieval]

    J --> M[Weather Info]
    J --> N[Market Prices]
    J --> O[Farming Advice]
    J --> P[Emergency Services]

    K --> Q[Speech Recognition]
    L --> R[Database Query]

    S[Response] --> T[Text-to-Speech]
    T --> U[Play Audio Response]
    U --> A

    V[Call Center Agent] --> W[Monitor Calls]
    W --> X[Access Call Logs]
    X --> Y[Database]
```

**Use Cases:**
- **UC-23:** Make Phone Call
- **UC-24:** Select Language
- **UC-25:** Navigate IVR Menu
- **UC-26:** Ask Voice Query
- **UC-27:** Receive Audio Response
- **UC-28:** Access Emergency Services

## Actor Definitions

### Primary Actors:
- **Farmer:** Main user of the system, seeks agricultural information and advice
- **Agricultural Expert:** Updates disease databases and treatment information
- **Extension Officer:** Monitors system usage and provides oversight

### Secondary Actors:
- **Google Speech-to-Text API:** Converts speech to text
- **Google Translate API:** Translates between languages
- **OpenWeatherMap API:** Provides weather data
- **Mobile Device:** Provides camera and GPS functionality
- **Phone System:** Enables IVR functionality

## System Boundaries

The AgriNathi system integrates with multiple external services:
- Cloud speech and translation services
- Weather data providers
- Mobile device capabilities
- Telephone networks
- Local databases for agricultural knowledge