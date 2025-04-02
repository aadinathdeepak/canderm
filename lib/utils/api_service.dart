import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:mime/mime.dart';
import 'package:http_parser/http_parser.dart';

class ApiService {
  static const String analyzeUrl =
      "http://127.0.0.1:5000/predict"; // Replace with actual API URL
  static const String reportUrl =
      "https://your-backend-api.com/send-report"; // Replace with actual API URL

  static Future<String> analyzeImage(File image) async {
    try {
      var uri = Uri.parse(analyzeUrl);
      var request = http.MultipartRequest("POST", uri);

      // Determine MIME type
      String mimeType = lookupMimeType(image.path) ?? "image/jpeg";

      request.files.add(
        await http.MultipartFile.fromPath(
          'file',
          image.path,
          contentType: MediaType.parse(mimeType),
        ),
      );

      var response = await request.send();

      if (response.statusCode == 200) {
        return await response.stream.bytesToString();
      } else {
        return "Error: Server responded with status code ${response.statusCode}";
      }
    } catch (e) {
      return "Error: Failed to analyze image - $e";
    }
  }

  static Future<bool> sendReport(File image) async {
    try {
      var uri = Uri.parse(reportUrl);
      var request = http.MultipartRequest("POST", uri);

      String mimeType = lookupMimeType(image.path) ?? "image/jpeg";

      request.files.add(
        await http.MultipartFile.fromPath(
          'file',
          image.path,
          contentType: MediaType.parse(mimeType),
        ),
      );

      var response = await request.send();

      return response.statusCode == 200;
    } catch (e) {
      print("Error sending report: $e");
      return false;
    }
  }
}
