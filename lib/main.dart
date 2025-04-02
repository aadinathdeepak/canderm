import 'package:canderm/pages/home_screen.dart';
import 'package:canderm/pages/login_screen.dart';
import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'firebase_options.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'dart:io' show Platform;

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );

  // ✅ Check if user is logged in using FirebaseAuth
  final User? user = FirebaseAuth.instance.currentUser;
  runApp(CanDermAI(isLoggedIn: user != null));
}

// ✅ Google Sign-In Configuration
final GoogleSignIn googleSignIn = GoogleSignIn(
  clientId: Platform.isIOS
      ? "96899027797-gbda8lbbr9blav5carlorqr9strh4trp.apps.googleusercontent.com" // iOS Client ID
      : Platform.isAndroid
          ? null // Android auto-fetches from google-services.json
          : "96899027797-hii09du2m47jgd3fgdf7j2dtogtqjmaf.apps.googleusercontent.com", // Web Client ID
  scopes: ['email'],
);

class CanDermAI extends StatelessWidget {
  final bool isLoggedIn;
  const CanDermAI({super.key, required this.isLoggedIn});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'CanDerm AI',
      theme: ThemeData.dark().copyWith(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.purple),
        scaffoldBackgroundColor: Colors.grey[900],
        appBarTheme: const AppBarTheme(
          backgroundColor: Colors.black,
          elevation: 8,
        ),
      ),
      debugShowCheckedModeBanner: false,
      home: isLoggedIn ? const HomeScreen() : const LoginScreen(),
    );
  }
}
