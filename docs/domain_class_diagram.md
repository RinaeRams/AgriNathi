# AgriNathi System - Domain Class Diagram

## Overview
The domain class diagram represents the key entities (things) in the AgriNathi agricultural voice assistant system. These classes represent the core business concepts and their relationships.

```mermaid
classDiagram
    class Farmer {
        +farmerId: String
        +name: String
        +phoneNumber: String
        +location: Location
        +language: String
        +farmSize: Float
        +cropTypes: List<Crop>
        +registrationDate: Date
        +queryHistory: List<Query>
        +getWeatherAdvice()
        +getDiseaseDiagnosis()
        +makeVoiceQuery()
    }

    class Location {
        +locationId: String
        +city: String
        +province: String
        +country: String
        +latitude: Float
        +longitude: Float
        +climateZone: String
        +getWeatherData()
        +getRegionalCrops()
    }

    class Query {
        +queryId: String
        +farmerId: String
        +queryType: QueryType
        +voiceData: Audio
        +textTranscript: String
        +translatedText: String
        +response: String
        +timestamp: DateTime
        +confidence: Float
        +processQuery()
        +generateResponse()
    }

    class Audio {
        +audioId: String
        +filePath: String
        +duration: Float
        +format: String
        +sampleRate: Integer
        +language: String
        +transcribe()
        +getDuration()
    }

    class WeatherData {
        +weatherId: String
        +location: Location
        +temperature: Float
        +humidity: Float
        +windSpeed: Float
        +precipitation: Float
        +description: String
        +forecast: List<Forecast>
        +timestamp: DateTime
        +getCurrentWeather()
        +getForecast()
        +generateFarmingAdvice()
    }

    class Forecast {
        +forecastId: String
        +date: Date
        +temperature: Float
        +description: String
        +precipitation: Float
        +humidity: Float
        +windSpeed: Float
    }

    class PlantImage {
        +imageId: String
        +farmerId: String
        +filePath: String
        +fileSize: Integer
        +format: String
        +uploadDate: DateTime
        +location: Location
        +analyzeImage()
        +getDiagnosis()
    }

    class Disease {
        +diseaseId: String
        +name: String
        +scientificName: String
        +affectedCrops: List<Crop>
        +symptoms: List<String>
        +causes: String
        +severity: SeverityLevel
        +treatment: List<Treatment>
        +prevention: List<String>
        +getTreatmentPlan()
        +getPreventionTips()
    }

    class Treatment {
        +treatmentId: String
        +diseaseId: String
        +description: String
        +method: TreatmentMethod
        +effectiveness: Float
        +duration: Integer
        +cost: Float
        +materials: List<String>
        +applyTreatment()
    }

    class Crop {
        +cropId: String
        +name: String
        +scientificName: String
        +season: Season
        +waterRequirement: WaterLevel
        +soilType: SoilType
        +commonDiseases: List<Disease>
        +marketPrice: Float
        +growthPeriod: Integer
        +getOptimalConditions()
        +getDiseaseRisk()
    }

    class MarketData {
        +marketId: String
        +crop: Crop
        +location: Location
        +price: Float
        +quantity: Float
        +date: Date
        +source: String
        +getAveragePrice()
        +getPriceTrend()
    }

    class AgriculturalAdvice {
        +adviceId: String
        +queryId: String
        +topic: AdviceTopic
        +content: String
        +priority: PriorityLevel
        +source: AdviceSource
        +validityPeriod: Integer
        +generateAdvice()
        +personalizeForFarmer()
    }

    class ExtensionOfficer {
        +officerId: String
        +name: String
        +region: String
        +specialization: String
        +contactInfo: String
        +assignedFarmers: List<Farmer>
        +monitorQueries()
        +provideExpertAdvice()
        +updateKnowledgeBase()
    }

    class SystemLog {
        +logId: String
        +timestamp: DateTime
        +userId: String
        +action: String
        +result: String
        +errorMessage: String
        +responseTime: Float
        +logActivity()
        +generateReports()
    }

    %% Relationships
    Farmer "1" --> "*" Query : makes
    Farmer "1" --> "1" Location : located in
    Farmer "1" --> "*" PlantImage : uploads
    Farmer "*" --> "1" ExtensionOfficer : supervised by

    Query "1" --> "1" Audio : contains
    Query "1" --> "1" AgriculturalAdvice : generates

    Location "1" --> "*" WeatherData : has
    Location "1" --> "*" MarketData : has

    WeatherData "1" --> "*" Forecast : contains

    PlantImage "1" --> "1" Disease : diagnosed as

    Disease "1" --> "*" Treatment : has
    Disease "*" --> "*" Crop : affects

    Crop "1" --> "*" MarketData : has prices for

    AgriculturalAdvice "1" --> "1" Query : responds to

    ExtensionOfficer "1" --> "*" Farmer : supervises

    SystemLog "*" --> "1" Farmer : logs activity for
    SystemLog "*" --> "1" Query : logs

    %% Enumerations
    class QueryType {
        <<enumeration>>
        WEATHER
        DISEASE_DIAGNOSIS
        FARMING_ADVICE
        MARKET_INFO
        GENERAL_INFO
    }

    class SeverityLevel {
        <<enumeration>>
        LOW
        MEDIUM
        HIGH
        CRITICAL
    }

    class TreatmentMethod {
        <<enumeration>>
        CHEMICAL
        ORGANIC
        CULTURAL
        BIOLOGICAL
        INTEGRATED
    }

    class Season {
        <<enumeration>>
        SPRING
        SUMMER
        AUTUMN
        WINTER
        YEAR_ROUND
    }

    class WaterLevel {
        <<enumeration>>
        LOW
        MEDIUM
        HIGH
        VERY_HIGH
    }

    class SoilType {
        <<enumeration>>
        CLAY
        SANDY
        LOAM
        SILT
        CHALK
    }

    class AdviceTopic {
        <<enumeration>>
        PLANTING
        IRRIGATION
        PEST_CONTROL
        DISEASE_MANAGEMENT
        HARVESTING
        WEATHER_ADAPTATION
        SOIL_MANAGEMENT
        MARKET_TIMING
    }

    class PriorityLevel {
        <<enumeration>>
        LOW
        MEDIUM
        HIGH
        URGENT
    }

    class AdviceSource {
        <<enumeration>>
        EXPERT_KNOWLEDGE
        RESEARCH_DATA
        LOCAL_EXPERIENCE
        WEATHER_DATA
        MARKET_ANALYSIS
    }
```

## Class Descriptions

### Core Entities

1. **Farmer**: Represents the primary user of the system
   - Contains personal and farm information
   - Interacts with all system modules
   - Maintains query history

2. **Location**: Geographic information for weather and regional advice
   - Supports location-based services
   - Links to weather and market data

3. **Query**: Voice/text queries from farmers
   - Central to the voice assistant functionality
   - Links audio input to responses

### Agricultural Domain

4. **Crop**: Agricultural products and their characteristics
   - Links diseases, treatments, and market data
   - Contains growing requirements

5. **Disease**: Plant diseases and their management
   - Comprehensive disease information
   - Treatment and prevention strategies

6. **Treatment**: Disease treatment methods
   - Multiple treatment approaches
   - Effectiveness and cost information

### Data and Services

7. **WeatherData**: Weather information and forecasts
   - Real-time and forecast data
   - Agricultural advice generation

8. **MarketData**: Agricultural market prices
   - Price tracking and trends
   - Location-specific pricing

9. **PlantImage**: Images for disease diagnosis
   - Image processing and analysis
   - Diagnosis results

### System Support

10. **ExtensionOfficer**: Agricultural experts
    - System monitoring and support
    - Knowledge base updates

11. **SystemLog**: System activity tracking
    - Usage analytics and reporting
    - Performance monitoring

## Key Relationships

- **Farmer** is the central entity connecting to all other classes
- **Location** provides geographic context for weather and market data
- **Query** represents the main interaction point for voice services
- **Disease** and **Treatment** form the knowledge base for plant health
- **Crop** connects agricultural products to diseases and market data
- **ExtensionOfficer** provides human oversight and expertise

## Design Principles

1. **Single Responsibility**: Each class has a clear, focused purpose
2. **High Cohesion**: Related attributes and methods are grouped together
3. **Low Coupling**: Classes are loosely connected through well-defined relationships
4. **Domain-Driven Design**: Classes reflect real-world agricultural concepts
5. **Extensibility**: Easy to add new crops, diseases, and treatments