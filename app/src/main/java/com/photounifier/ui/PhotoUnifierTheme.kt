package com.photounifier.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val LightColors = lightColorScheme(
    primary = Color(0xFF365DA8),
    onPrimary = Color.White,
    primaryContainer = Color(0xFFDCE5FF),
    onPrimaryContainer = Color(0xFF071A43),
    secondary = Color(0xFF555F71),
    secondaryContainer = Color(0xFFDEE3F2),
    onSecondaryContainer = Color(0xFF171D2B),
    background = Color(0xFFF6F7FB),
    surface = Color(0xFFFFFFFF),
    surfaceVariant = Color(0xFFE8EAF2),
    onSurface = Color(0xFF191B22),
    onSurfaceVariant = Color(0xFF444750)
)

private val DarkColors = darkColorScheme(
    primary = Color(0xFFB4C7FF),
    onPrimary = Color(0xFF17316B),
    primaryContainer = Color(0xFF2C467F),
    onPrimaryContainer = Color(0xFFDCE5FF),
    secondary = Color(0xFFC0C7DA),
    secondaryContainer = Color(0xFF3D4558),
    onSecondaryContainer = Color(0xFFE0E5F8),
    background = Color(0xFF111318),
    surface = Color(0xFF191B22),
    surfaceVariant = Color(0xFF30333D),
    onSurface = Color(0xFFE2E2EA),
    onSurfaceVariant = Color(0xFFC4C6D0)
)

@Composable
fun PhotoUnifierTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = if (isSystemInDarkTheme()) DarkColors else LightColors,
        content = content
    )
}
