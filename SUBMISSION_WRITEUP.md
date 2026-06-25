# Capstone Pitch & Submission Writeup

**Project Title:** Ambient Wildlife Poaching Guardian  
**Track:** Freestyle Track  
**Author:** Paula Andrea  
**Technologies:** Google Agent Development Kit (ADK 2.0), FastAPI, Gemini 3.5 Flash, Live Open-Meteo & USGS NWIS APIs, SMART Patrol MCP API

---

## 1. Executive Summary & Pitch

### Problem Statement
Vast, remote forest reserves are highly vulnerable to illegal poaching. While reserves deploy acoustic and camera-trap sensors, they produce thousands of hours of raw audio. Manual monitoring is impossible, and simple threshold alarms generate high false-alarm rates (caused by wind, thunder, and safe wildlife calls), leading to operator alert fatigue and missed poaching events.

### The Solution: Ambient Wildlife Poaching Guardian
Instead of a single-turn chatbot, we built a **Resilient Anti-Poaching Operations Command Fabric (Common Operational Picture)**. Built on ADK 2.0, the guardian ingests telemetry and orchestrates a pipeline of specialized agents:
1. **Sensor Intake Agent:** Normalizes telemetry into CloudEvent envelopes.
2. **Acoustic Triage Agent:** Pre-filters harmless wildlife vocalizations below 50dB to conserve computing resources.
3. **Context Fusion Agent:** Simulates Model Context Protocol (MCP) tool lookups to merge weather forecast variables, scheduled ranger GPS patrol routes, and population density indices.
4. **Threat Assessor Agent (SafeLlmAgent):** Evaluates anomalies alongside the fused context to output a structured threat object (with confidence scores, recommended actions, and a top-evidence list).
5. **Policy & Safety Agent:** Applies deterministic rules (e.g. overrides drone launches if wind > 40 km/h; restricts altitudes near Fern Canyon breeding corridors; blocks wading patrols to Tall Trees Grove if Redwood Creek streamflow exceeds flow thresholds).
6. **Dispatcher Agent:** coordinates drone flight vectors and ranger radio frequencies upon confirmation.
7. **After-Action Agent:** Compiles post-incident timeline report logs.

### Why Agents & ADK?
- **Stateful Resumption:** The agent must safely pause and wait hours for ranger confirmation (`RequestInput`) without losing sensor or threat context.
- **Dynamic Context-Fusion Triage:** Simple classifiers fail. Combining acoustics with patrol roster schedules and weather indices via MCP tools allows the agent to make intelligent recommendations (e.g., divert rangers instead of flying drones in high winds).
- **Edge Resilience:** Supports Primary IP link, Sat-Link CBOR compression, or off-grid LoRa Mesh Meshtastic fallback.

---

## 2. YouTube Video Script Outline (5-Minute Limit)

| Timestamp | Section | Key Visuals | Voiceover Script Summary |
| :--- | :--- | :--- | :--- |
| **0:00 - 0:45** | **The Problem** | B-Roll of rainforests, rangers looking at screens. | "Remote reserves are vast. Poachers exploit the gaps. Rangers are overwhelmed by sensor data. Alert fatigue is a silent killer." |
| **0:45 - 1:30** | **Why Agents?** | Mermaid Architecture Diagram. | "We built an ADK 2.0 event-driven agent. It doesn't just prompt; it orchestrates checks, handles states, and executes decisions." |
| **1:30 - 2:30** | **The Architecture** | Diagram showing Ingest -> MCP -> LLM -> HITL. | "Demonstrating three key concepts: 1) ADK 2.0 Graph Workflow, 2) live weather & hydrology context-fusion, 3) Secure HITL." |
| **2:30 - 4:00** | **Dashboard Demo** | Screensharing of the localhost:8080 dashboard. | "Let's trigger a gunshot. The threat assessment is processed. The Ranger dashboard flashes a critical alert. I click Deploy Drone, and the agent completes." |
| **4:00 - 5:00** | **The Build & Close** | Showing Makefile, safe fallback code. | "Created using Google Agents CLI. 100% cost-safe local verification framework. Ready for global conservation deployment." |

---

## 3. Written Submission & Project Journey

### Scaffolding with Conductor & Conductor ADK
We initialized the project using `agents-cli scaffold` inside our Antigravity environment. The ADK 2.0 graph API was leveraged to design a modular network of nodes and conditional edges.

### The Offline Safe-Fallback Innovation
A major breakthrough during development was implementing a `SafeLlmAgent` subclass. If reviewers or rangers run this agent locally without Vertex AI credentials, the workflow detects it and falls back to mock intelligence. This guarantees **$0 cloud cost** and offline capability.

### Secure Human-in-the-Loop (HITL) Implementation
Using ADK's `RequestInput` yield, the agent pauses when a high-threat incident occurs. The session rehydration and resume flows are managed asynchronously in the FastAPI backend by passing `types.FunctionResponse` objects back into the ADK runner.
