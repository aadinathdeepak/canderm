import 'package:flutter/material.dart';

class NotebookBackgroundPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.deepPurple.shade700
      ..style = PaintingStyle.fill;

    // Background fill
    canvas.drawRect(Rect.fromLTWH(0, 0, size.width, size.height), paint);

    // Draw horizontal notebook lines
    final linePaint = Paint()
      ..color = Colors.white.withOpacity(0.1)
      ..strokeWidth = 2;

    const double lineSpacing = 40.0;
    for (double y = lineSpacing; y < size.height; y += lineSpacing) {
      canvas.drawLine(Offset(0, y), Offset(size.width, y), linePaint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
