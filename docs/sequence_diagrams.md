# AgriNathi System - Sequence Diagrams

## Overview
This document contains simplified sequence diagrams for key use cases in the AgriNathi agricultural voice assistant system. Each diagram shows the interaction flow between actors and system components.

## 1. Voice Query Processing (UC-1 to UC-5)

```mermaid
sequenceDiagram
    participant F as Farmer
    participant UI as Web Interface
    participant VR as Voice Recognition Service
    participant STT as Google Speech-to-Text
    participant GT as Google Translate
    participant AE as Advice Engine
    participant DB as Database

    F->>UI: Click voice record button
    UI->>UI: Start audio recording
    F->>UI: Speak isiZulu query
    UI->>UI: Stop recording
    UI->>VR: Send audio data
    VR->>STT: Transcribe audio (zu-ZA)
    STT-->>VR: Return transcript
    VR->>GT: Translate to English
    GT-->>VR: Return translation
    VR->>AE: Generate advice
    AE->>DB: Query knowledge base
    DB-->>AE: Return relevant advice
    AE-->>VR: Return personalized advice
    VR-->>UI: Return response
    UI-->>F: Display transcript, translation & advice
    UI->>DB: Log query & response
```

## 2. Weather Information Retrieval (UC-6 to UC-10)

```mermaid
sequenceDiagram
    participant F as Farmer
    participant UI as Weather Interface
    participant WS as Weather Service
    participant OWM as OpenWeatherMap API
    participant CACHE as Cache System
    participant AE as Advice Engine

    F->>UI: Enter/select location
    UI->>WS: Request weather data
    WS->>CACHE: Check cached data
    CACHE-->>WS: Return cached data (if fresh)
    WS->>OWM: Fetch current weather
    OWM-->>WS: Return current weather
    WS->>OWM: Fetch 5-day forecast
    OWM-->>WS: Return forecast data
    WS->>CACHE: Store weather data
    WS->>AE: Generate farming advice
    AE-->>WS: Return advice
    WS-->>UI: Return weather + advice
    UI-->>F: Display weather info & advice
```

## 3. Plant Disease Diagnosis (UC-11 to UC-16)

```mermaid
sequenceDiagram
    participant F as Farmer
    participant UI as Plant Scanner Interface
    participant PS as Plant Scanner Service
    participant IP as Image Processor
    participant ML as ML Disease Model
    participant DB as Disease Database
    participant TE as Treatment Engine

    F->>UI: Upload/capture plant image
    UI->>PS: Send image file
    PS->>IP: Preprocess image
    IP-->>PS: Return processed image
    PS->>ML: Analyze for diseases
    ML->>DB: Query disease patterns
    DB-->>ML: Return disease data
    ML-->>PS: Return diagnosis results
    PS->>TE: Get treatment plan
    TE->>DB: Query treatments
    DB-->>TE: Return treatment options
    TE-->>PS: Return treatment plan
    PS-->>UI: Return diagnosis & treatment
    UI-->>F: Display results & recommendations
    PS->>DB: Log diagnosis
```

## 4. Mobile App Voice Assistant (UC-17 to UC-22)

```mermaid
sequenceDiagram
    participant F as Farmer
    participant MA as Mobile App
    participant VS as Voice Service
    participant STT as Speech-to-Text
    participant GT as Translate
    participant BE as Backend API
    participant DB as Database

    F->>MA: Tap voice assistant button
    MA->>MA: Request microphone permission
    MA->>MA: Start voice recording
    F->>MA: Speak query
    MA->>MA: Stop recording
    MA->>VS: Send audio data
    VS->>STT: Transcribe speech
    STT-->>VS: Return text
    VS->>GT: Translate if needed
    GT-->>VS: Return translation
    VS->>BE: Send query to backend
    BE->>DB: Query knowledge base
    DB-->>BE: Return response
    BE-->>VS: Return advice
    VS-->>MA: Return response
    MA-->>F: Display/play response
    MA->>DB: Cache response (offline)
```

## 5. IVR System Interaction (UC-23 to UC-28)

```mermaid
sequenceDiagram
    participant F as Farmer
    participant TEL as Telephone Network
    participant IVR as IVR System
    participant TTS as Text-to-Speech Engine
    participant DB as Database
    participant LOG as Call Logger

    F->>TEL: Dial AgriNathi number
    TEL->>IVR: Connect call
    IVR->>IVR: Play welcome message
    IVR->>TTS: Convert to speech
    TTS-->>IVR: Return audio
    IVR-->>F: "Welcome to AgriNathi"
    IVR->>F: Request language selection
    F->>IVR: Press key for isiZulu
    IVR->>IVR: Set language to zu-ZA
    IVR->>F: Play menu options
    F->>IVR: Press key for weather
    IVR->>DB: Query weather data
    DB-->>IVR: Return weather info
    IVR->>TTS: Convert response to speech
    TTS-->>IVR: Return audio
    IVR-->>F: Play weather information
    IVR->>LOG: Log call details
    LOG->>DB: Store call log
```

## 6. System Monitoring and Analytics

```mermaid
sequenceDiagram
    participant EO as Extension Officer
    participant UI as Admin Interface
    participant AS as Analytics Service
    participant DB as Database
    participant LOG as System Logs

    EO->>UI: Login to admin panel
    UI->>AS: Request dashboard data
    AS->>DB: Query usage statistics
    DB-->>AS: Return query counts
    AS->>LOG: Get system performance
    LOG-->>AS: Return performance metrics
    AS->>DB: Get farmer demographics
    DB-->>AS: Return user data
    AS-->>UI: Return analytics data
    UI-->>EO: Display dashboard
    EO->>UI: Export reports
    UI->>AS: Generate report
    AS->>DB: Compile report data
    DB-->>AS: Return compiled data
    AS-->>UI: Return report file
    UI-->>EO: Download report
```

## 7. Knowledge Base Update Process

```mermaid
sequenceDiagram
    participant AE as Agricultural Expert
    participant UI as Admin Interface
    participant KB as Knowledge Base Service
    participant DB as Disease Database
    participant VAL as Validation Service
    participant NOTIF as Notification Service

    AE->>UI: Login to expert panel
    UI->>KB: Request disease database
    KB->>DB: Query current diseases
    DB-->>KB: Return disease list
    KB-->>UI: Display diseases
    AE->>UI: Add new disease
    UI->>KB: Submit new disease data
    KB->>VAL: Validate disease data
    VAL-->>KB: Return validation result
    KB->>DB: Insert new disease
    DB-->>KB: Confirm insertion
    KB->>NOTIF: Notify system update
    NOTIF->>UI: Show success message
    UI-->>AE: Display confirmation
```

## 8. Emergency Services Access

```mermaid
sequenceDiagram
    participant F as Farmer
    participant UI as Emergency Interface
    participant ES as Emergency Service
    participant DB as Emergency Contacts
    participant TEL as Telephone System
    participant LOG as Emergency Logger

    F->>UI: Access emergency services
    UI->>ES: Request emergency options
    ES->>DB: Query emergency contacts
    DB-->>ES: Return contact list
    ES-->>UI: Display emergency options
    F->>UI: Select veterinary emergency
    UI->>ES: Initiate emergency call
    ES->>TEL: Dial emergency number
    TEL-->>ES: Call connected
    ES->>LOG: Log emergency call
    LOG->>DB: Store emergency log
    ES-->>F: Call in progress
    TEL->>F: Connect to emergency service
```

## Common Patterns

### Error Handling Pattern
```mermaid
sequenceDiagram
    participant U as User
    participant S as Service
    participant API as External API
    participant FB as Fallback Service

    U->>S: Make request
    S->>API: Call external service
    API-->>S: Error response
    S->>FB: Try fallback
    FB-->>S: Fallback response
    S-->>U: Return fallback data
    S->>LOG: Log error
```

### Caching Pattern
```mermaid
sequenceDiagram
    participant S as Service
    participant CACHE as Cache
    participant API as External API

    S->>CACHE: Check cache
    CACHE-->>S: Cache hit
    S-->>S: Return cached data
    Note over S: Skip API call
    S->>CACHE: Check cache
    CACHE-->>S: Cache miss
    S->>API: Fetch fresh data
    API-->>S: Return data
    S->>CACHE: Store in cache
    S-->>S: Return fresh data
```

### Authentication Pattern
```mermaid
sequenceDiagram
    participant U as User
    participant UI as Interface
    participant AUTH as Auth Service
    participant DB as User Database

    U->>UI: Login request
    UI->>AUTH: Validate credentials
    AUTH->>DB: Query user data
    DB-->>AUTH: Return user info
    AUTH->>AUTH: Verify password
    AUTH-->>UI: Return auth token
    UI-->>U: Login successful
    U->>UI: Access protected resource
    UI->>AUTH: Validate token
    AUTH-->>UI: Token valid
    UI-->>U: Grant access
```

## Performance Considerations

1. **Caching**: Weather data cached for 30 minutes
2. **Async Processing**: Image analysis runs asynchronously
3. **Database Indexing**: Query logs indexed by timestamp and user
4. **CDN**: Static assets served via CDN
5. **Load Balancing**: API calls distributed across multiple servers

## Security Measures

1. **Input Validation**: All user inputs validated
2. **Rate Limiting**: API calls limited per user
3. **Encryption**: Sensitive data encrypted in transit and at rest
4. **Audit Logging**: All system interactions logged
5. **Access Control**: Role-based permissions for admin functions