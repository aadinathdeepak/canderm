import 'dart:io';
import 'package:canderm/pages/nearby_vets_screen.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:percent_indicator/circular_percent_indicator.dart';
import '../utils/api_service.dart';
import '../widgets/custom_button.dart';
import 'dart:convert';

class ResultScreen extends StatelessWidget {
  final File image;
  final String result;

  const ResultScreen({super.key, required this.image, required this.result});

  void _sendReport(BuildContext context) async {
    bool success = await ApiService.sendReport(image);

    if (!context.mounted) return;

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content:
            Text(success ? "Report sent to vet!" : "Failed to send report"),
        backgroundColor: success ? Colors.green : Colors.red,
      ),
    );
  }

  String _formatResult(String result) {
    try {
      // Convert JSON-like string to a Map
      result = result.replaceAll("'", "\""); // Ensure valid JSON format
      Map<String, dynamic> parsedResult =
          Map<String, dynamic>.from(jsonDecode(result));

      // Format key-value pairs as readable text
      return parsedResult.entries
          .map((entry) => "${entry.key}: ${entry.value}")
          .join("\n");
    } catch (e) {
      print("Error formatting result: $e");
      return result; // Return original if parsing fails
    }
  }

  @override
  Widget build(BuildContext context) {
    double severityPercentage = _extractSeverityPercentage(result);

    double confidenceScore = _extractConfidenceScore(result);

    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Analysis Result',
          style: GoogleFonts.rubikDoodleShadow(
            fontSize: 28,
            fontWeight: FontWeight.bold,
            color: Colors.amber,
          ),
        ),
        centerTitle: true,
        backgroundColor: Colors.deepPurple.shade900,
        elevation: 10,
      ),
      backgroundColor: Colors.deepPurple.shade800,
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // 📌 Display Image
          ClipRRect(
            borderRadius: BorderRadius.circular(16),
            child: Image.file(
              image,
              width: double.infinity,
              height: 280,
              fit: BoxFit.cover,
            ),
          ),
          const SizedBox(height: 20),

          // 📌 Disease Classification Card
          Card(
            color: Colors.deepPurple.shade700,
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            elevation: 4,
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Center(
                    child: const Text(
                      'Disease Analysis',
                      style: TextStyle(
                        fontSize: 22,
                        fontWeight: FontWeight.bold,
                        color: Colors.white70,
                      ),
                    ),
                  ),
                  const SizedBox(height: 10),
                  Center(
                    child: Text(
                      _formatResult(result),
                      textAlign: TextAlign.left,
                      style: const TextStyle(
                        fontSize: 18,
                        color: Colors.white60,
                        fontWeight: FontWeight.w300,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),

          const SizedBox(height: 20),

          // 📌 Severity Score - Circular Progress Indicator
          Card(
            color: Colors.deepPurple.shade700,
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            elevation: 4,
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  const Text(
                    'Affected Area Percentage',
                    style: TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                      color: Colors.white70,
                    ),
                  ),
                  const SizedBox(height: 20),
                  CircularPercentIndicator(
                    radius: 70.0,
                    lineWidth: 13.0,
                    animation: true,
                    animationDuration: 750,
                    percent: severityPercentage / 100, // Convert to fraction
                    center: Text(
                      "${severityPercentage.toStringAsFixed(1)}%",
                      style: const TextStyle(
                        fontSize: 22,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                    circularStrokeCap: CircularStrokeCap.round,
                    progressColor: severityPercentage > 60
                        ? Colors.red
                        : (severityPercentage > 30
                            ? Colors.orange
                            : Colors.green),
                    backgroundColor: Colors.white24,
                  ),
                ],
              ),
            ),
          ),

          const SizedBox(height: 20),

          // 📌 Confidence Score - Circular Progress Indicator
          Card(
            color: Colors.deepPurple.shade700,
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            elevation: 4,
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  const Text(
                    'Confidence Score',
                    style: TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                      color: Colors.white70,
                    ),
                  ),
                  const SizedBox(height: 20),
                  CircularPercentIndicator(
                    radius: 70.0,
                    lineWidth: 13.0,
                    animation: true,
                    animationDuration: 1000,
                    percent: confidenceScore / 100, // Convert to fraction
                    center: Text(
                      "${confidenceScore.toStringAsFixed(1)}%",
                      style: const TextStyle(
                        fontSize: 22,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                    circularStrokeCap: CircularStrokeCap.round,
                    progressColor: confidenceScore > 75
                        ? Colors.green
                        : (confidenceScore > 50 ? Colors.orange : Colors.red),
                    backgroundColor: Colors.white24,
                  ),
                ],
              ),
            ),
          ),

          const SizedBox(height: 20),

          // 📌 Send Report Button
          CustomButton(
            text: "Send Report to Vet",
            onPressed: () {
              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (context) =>
                      NearbyVetsScreen(image: image, result: result),
                ),
              );
            },
          ),
        ],
      ),
    );
  }

  // ✅ Extracts severity percentage from result (assumes response contains it)
  double _extractSeverityPercentage(String result) {
    try {
      RegExp regExp = RegExp(r"(\d+(\.\d+)?)%");
      Match? match = regExp.firstMatch(result);
      if (match != null) {
        return double.parse(match.group(1)!);
      }
    } catch (e) {
      print("Error parsing severity: $e");
    }
    return 0.0; // Default if extraction fails
  }

  double _extractConfidenceScore(String result) {
    try {
      print("Raw result: $result"); // Debugging statement

      // Convert JSON-like string to a Map
      result = result.replaceAll("'", "\""); // Ensure valid JSON format
      Map<String, dynamic> parsedResult =
          Map<String, dynamic>.from(jsonDecode(result));

      // Extract confidence score
      if (parsedResult.containsKey("Confidence Score")) {
        String confidenceString = parsedResult["Confidence Score"];
        print(
            "Extracted Confidence Score String: $confidenceString"); // Debugging

        // Remove the percentage sign and convert to double
        return double.parse(confidenceString.replaceAll("%", ""));
      } else {
        print("Key 'Confidence Score' not found!"); // Debugging
      }
    } catch (e) {
      print("Error parsing confidence score: $e");
    }

    return 0.0; // Default if extraction fails
  }
}
