# SignalSafe AI 

## Team Members: 

Landen Shurtz- lpshurtz@asu.edu,
Edward Hardiman- ejhardim@asu.edu,
Trent Mccann- tmmccann@asu.edu,
Parker Wakefield- pwakefi1@asu.edu

The primary user of this product is a drone operator using an ArduPilot-style system for inspection, mapping, surveying, research, or recreational flight. Today, these users often rely on reactive failsafe behavior after communication quality has already degraded or failed, leaving them with limited time to respond safely.

This problem matters over the next 3–5 years because drones are increasingly used for mapping, inspection, agriculture, and research, making communication reliability more important. Our product is designed to detect rising communication-loss risk before the connection is fully lost. It would analyze telemetry indicators, warn the operator when signal-loss risk is increasing, and recommend a response such as continuing monitoring, hovering, returning to home, or manual takeover. AI adds value because it can analyze multiple inputs at once, such as signal strength, packet loss, latency, GPS quality, battery level, altitude, and speed, to recognize more complex warning patterns than a simple threshold-only system.

For the technical concept, we would use drone telemetry data, including signal strength, packet loss, latency, GPS quality, battery level, altitude, and speed. The idea is to use those values to spot patterns that could signal that loss is becoming more likely. For that part, we could use a basic classifier or anomaly-detection model. The nanoGPT component would primarily be used to generate short explanations that tell the operator why the system thinks the flight is risky and what action should be taken next.

A realistic MVP for this course would be a simulated prototype that uses simulated drone telemetry data to predict communication-loss risk and recommend the next course of action for the operator. In version one, a user can upload a simulated telemetry log, and the system returns a risk score, a suggested action, and a short explanation of the result.

The biggest three unknowns right now are whether we can get enough realistic telemetry data, how we would test whether the recommendations are helpful without a live drone setup, and whether drone operators would trust an AI tool like this. For our resources, the group will use synthetic telemetry data and simulated communication-loss cases, then later expand to public datasets from Kaggle, Hugging Face, or other open drone and robotics sources.

## Video Pitch 

Watch the the SignalSafe AI pitch: https://youtu.be/H_unND9jSoA?si=MEKJlEpDNz9mt0l7
