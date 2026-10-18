package com.photounifier

/** The processing route selected for one image. */
enum class ImageKind(val title: String, val labelRes: Int) {
    DOCUMENT("مستند", R.string.kind_document),
    POSTER("بوستر", R.string.kind_poster),
    INTERIOR("مكان داخلي", R.string.kind_interior),
    EXTERIOR("مكان خارجي", R.string.kind_exterior),
    GENERAL("صورة عامة", R.string.kind_general)
}
