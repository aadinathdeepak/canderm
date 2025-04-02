import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:canderm/main.dart';

void main() {
  testWidgets('Check if Upload Button is Present', (WidgetTester tester) async {
    // Build the app and trigger a frame.
    await tester.pumpWidget(
      const MaterialApp(
        home: CanDermAI(isLoggedIn: false), // Provide 'isLoggedIn' parameter
      ),
    );

    // Verify that the upload button text is present in the UI.
    expect(find.text('Upload Dog Skin Image'), findsOneWidget);
  });
}
