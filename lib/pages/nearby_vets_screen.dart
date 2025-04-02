import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:geolocator/geolocator.dart';
import 'package:http/http.dart' as http;
import 'package:pdf/widgets.dart' as pw;
import 'package:path_provider/path_provider.dart';
import 'package:open_file/open_file.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:url_launcher/url_launcher.dart';
import 'dart:typed_data';
import 'package:pdf/pdf.dart';

class NearbyVetsScreen extends StatefulWidget {
  final File image;
  final String result;

  const NearbyVetsScreen(
      {super.key, required this.image, required this.result});

  @override
  State<NearbyVetsScreen> createState() => _NearbyVetsScreenState();
}

class _NearbyVetsScreenState extends State<NearbyVetsScreen> {
  bool _loading = true;
  List<Map<String, dynamic>> _nearbyVets = [];
  late Position _currentPosition;
  Map<String, dynamic> _userProfile = {};
  final String _apiKey = "AIzaSyB6pMd1Qgar2KKGOvgf9m12sry2SF7Zp3Q";

  @override
  void initState() {
    super.initState();
    _fetchLocationAndVets();
    _loadUserProfile();
  }

  Future<void> _loadUserProfile() async {
    final user = FirebaseAuth.instance.currentUser;
    if (user != null) {
      final doc = await FirebaseFirestore.instance
          .collection('users')
          .doc(user.uid)
          .get();
      if (doc.exists) {
        setState(() {
          _userProfile = doc.data()!;
        });
      }
    }
  }

  Future<void> _fetchLocationAndVets() async {
    bool serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) {
      ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("Location services are disabled.")));
      return;
    }

    LocationPermission permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.denied) {
        ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text("Location permission denied.")));
        return;
      }
    }

    if (permission == LocationPermission.deniedForever) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
          content: Text("Location permissions are permanently denied.")));
      return;
    }

    _currentPosition = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high);
    final lat = _currentPosition.latitude;
    final lng = _currentPosition.longitude;

    final url = Uri.parse(
        "https://maps.googleapis.com/maps/api/place/nearbysearch/json?location=$lat,$lng&radius=15000&type=veterinary_care&key=$_apiKey");
    final response = await http.get(url);

    if (response.statusCode == 200) {
      final data = json.decode(response.body);
      final results = data['results'] as List;

      _nearbyVets = results.take(7).map((place) {
        final name = place['name'];
        final lat = place['geometry']['location']['lat'];
        final lng = place['geometry']['location']['lng'];
        final distance = Geolocator.distanceBetween(_currentPosition.latitude,
                _currentPosition.longitude, lat, lng) /
            1000.0;

        return {
          'name': name,
          'lat': lat,
          'lng': lng,
          'distance': distance.toStringAsFixed(2)
        };
      }).toList();
    } else {
      _nearbyVets = [];
    }

    setState(() => _loading = false);
  }

  Future<void> _sendReportToVet(String vetName) async {
    final pdf = pw.Document(); // Creating a new PDF document

    // Read the image file and convert it to Uint8List format
    final Uint8List imageBytes = await widget.image.readAsBytes();
    final pw.MemoryImage pdfImage = pw.MemoryImage(imageBytes);

    pdf.addPage(
      pw.Page(
        margin: pw.EdgeInsets.all(30),
        build: (pw.Context context) => pw.Column(
          crossAxisAlignment: pw.CrossAxisAlignment.start,
          children: [
            // **Header with Title**
            pw.Row(
              mainAxisAlignment: pw.MainAxisAlignment.spaceBetween,
              children: [
                pw.Text("CanDerm AI - Analysis Report",
                    style: pw.TextStyle(
                      fontSize: 22,
                      fontWeight: pw.FontWeight.bold,
                    )),
                // pw.Container(
                //   width: 50,
                //   height: 50,
                //   decoration: pw.BoxDecoration(
                //     shape: pw.BoxShape.circle,
                //     border: pw.Border.all(width: 2),
                //   ),
                //   child: pw.Center(
                //     child: pw.Text("CD", style: pw.TextStyle(fontSize: 30)),
                //   ),
                // ),
              ],
            ),
            pw.Divider(thickness: 2),
            pw.SizedBox(height: 15),

            // **Vet & Pet Owner Information**
            pw.Text("Veterinary Clinic: $vetName",
                style:
                    pw.TextStyle(fontSize: 16, fontWeight: pw.FontWeight.bold)),
            pw.SizedBox(height: 10),

            pw.Container(
              padding: pw.EdgeInsets.all(10),
              decoration: pw.BoxDecoration(
                border: pw.Border.all(width: 1),
                borderRadius: pw.BorderRadius.circular(5),
              ),
              child: pw.Column(
                crossAxisAlignment: pw.CrossAxisAlignment.start,
                children: [
                  pw.Text("Pet Owner Details",
                      style: pw.TextStyle(
                          fontSize: 16, fontWeight: pw.FontWeight.bold)),
                  pw.SizedBox(height: 5),
                  pw.Text("Name: ${_userProfile['name'] ?? 'N/A'}"),
                  pw.Text("Phone: ${_userProfile['phone'] ?? 'N/A'}"),
                  pw.Text("Pet Name: ${_userProfile['petName'] ?? 'N/A'}"),
                  pw.Text("Breed: ${_userProfile['breed'] ?? 'N/A'}"),
                ],
              ),
            ),
            pw.SizedBox(height: 15),

            // **Diagnosis & Analysis Result**
            pw.Container(
              padding:
                  pw.EdgeInsets.all(10), // Same padding as Pet Owner Details
              decoration: pw.BoxDecoration(
                border: pw.Border.all(width: 1), // Same border
                borderRadius: pw.BorderRadius.circular(5), // Same border radius
              ),
              child: pw.Column(
                crossAxisAlignment: pw.CrossAxisAlignment.start,
                children: [
                  pw.Text(
                    "Diagnosis Result",
                    style: pw.TextStyle(
                        fontSize: 16, fontWeight: pw.FontWeight.bold),
                  ),
                  pw.SizedBox(height: 5),
                  pw.Text(
                    widget.result.replaceAll(RegExp(r'[\[\]{}"]'),
                        ''), // Removes brackets and quotes
                  ),

                  // Matches the pet owner details format
                ],
              ),
            ),

            pw.SizedBox(height: 20),

            // **Image with Border**
            pw.Text("Analyzed Image",
                style:
                    pw.TextStyle(fontSize: 16, fontWeight: pw.FontWeight.bold)),
            pw.SizedBox(height: 15),
            pw.Container(
              width: 300,
              height: 250,
              decoration: pw.BoxDecoration(
                border: pw.Border.all(width: 2),
                borderRadius: pw.BorderRadius.circular(5),
              ),
              child: pw.Center(
                child: pw.Image(pdfImage, width: 250, height: 250),
              ),
            ),
            pw.SizedBox(height: 20),

            // **Footer with Signature Line**
            pw.Divider(thickness: 2),
            pw.SizedBox(height: 10),
            pw.Text("Doctor's Signature:",
                style:
                    pw.TextStyle(fontSize: 14, fontWeight: pw.FontWeight.bold)),
            pw.SizedBox(height: 30),
            pw.Text("_______________________",
                style: pw.TextStyle(fontSize: 16)),
            pw.Text("Vet Name & Stamp", style: pw.TextStyle(fontSize: 12)),
          ],
        ),
      ),
    );

    final output = await getTemporaryDirectory();
    final file = File("${output.path}/canine_report.pdf");
    await file.writeAsBytes(await pdf.save()); // Saving the PDF file

    ScaffoldMessenger.of(context)
        .showSnackBar(SnackBar(content: Text("Report sent to $vetName")));
    await OpenFile.open(file.path); // Opening the PDF file
  }

  void _navigateToVet(double lat, double lng) async {
    final url = "https://www.google.com/maps/dir/?api=1&destination=$lat,$lng";
    if (await canLaunch(url)) {
      await launch(url);
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("Could not open Google Maps.")));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Nearby Vets",
            style: GoogleFonts.rubikDoodleShadow(
              fontSize: 30,
              color: Colors.amber,
              fontWeight: FontWeight.bold,
            )),
        backgroundColor: Colors.deepPurple.shade900,
        centerTitle: true,
      ),
      backgroundColor: Colors.deepPurple.shade800,
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : ListView.builder(
              padding: const EdgeInsets.all(16),
              itemCount: _nearbyVets.length,
              itemBuilder: (context, index) {
                final vet = _nearbyVets[index];
                return Card(
                  color: Colors.deepPurple.shade700,
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(14)),
                  margin: const EdgeInsets.only(bottom: 16),
                  child: Padding(
                    padding: const EdgeInsets.all(12.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(vet['name'],
                            style: const TextStyle(
                                color: Colors.white,
                                fontWeight: FontWeight.w600,
                                fontSize: 18)),
                        const SizedBox(height: 4),
                        Text("Distance: ${vet['distance']} km",
                            style: const TextStyle(color: Colors.white70)),
                        const SizedBox(height: 10),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            ElevatedButton.icon(
                              onPressed: () => _sendReportToVet(vet['name']),
                              icon: const Icon(Icons.picture_as_pdf),
                              label: const Text("Send Report"),
                              style: ElevatedButton.styleFrom(
                                foregroundColor: Colors.black,
                                backgroundColor: Colors.amber,
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(10),
                                ),
                              ),
                            ),
                            ElevatedButton.icon(
                              onPressed: () =>
                                  _navigateToVet(vet['lat'], vet['lng']),
                              icon: const Icon(Icons.navigation),
                              label: const Text("Navigate"),
                              style: ElevatedButton.styleFrom(
                                foregroundColor: Colors.white,
                                backgroundColor: Colors.deepPurple.shade900,
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(10),
                                ),
                              ),
                            ),
                          ],
                        )
                      ],
                    ),
                  ),
                );
              },
            ),
    );
  }
}
