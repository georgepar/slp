"""Feature configurations for multimodal datasets

This configuration file contains all the possible combinations of
preprocessed POM, MOSI & MOSEI data, as extracted by the CMU team.

* text: `[glove | raw words]`
* audio: `[covarep | opensmile (for mosi)]`
* visual: `[facet | openface]`
"""

# MOSI

MOSI_OPENSMILE_FEATS = "CMU_MOSI_openSMILE_IS09"
# MOSI_OPENSMILE_FEATS = CMU_MOSI_openSMILE_EB10

MOSI_COVAREP_FACET_RAW = {
    "audio": "CMU_MOSI_COVAREP",
    "text": "CMU_MOSI_TimestampedWords",
    "raw": "CMU_MOSI_TimestampedWords",
    "visual": "CMU_MOSI_Visual_Facet_42",
    "labels": "CMU_MOSI_Opinion_Labels",
}

MOSI_COVAREP_FACET_GLOVE = {
    "audio": "CMU_MOSI_COVAREP",
    "text": "CMU_MOSI_TimestampedWordVectors",
    "raw": "CMU_MOSI_TimestampedWords",
    "visual": "CMU_MOSI_Visual_Facet_42",
    "labels": "CMU_MOSI_Opinion_Labels",
}

MOSI_COVAREP_OPENFACE_RAW = {
    "audio": "CMU_MOSI_COVAREP",
    "text": "CMU_MOSI_TimestampedWords",
    "raw": "CMU_MOSI_TimestampedWords",
    "visual": "CMU_MOSI_Visual_OpenFace_2",
    "labels": "CMU_MOSI_Opinion_Labels",
}

MOSI_COVAREP_OPENFACE_GLOVE = {
    "audio": "CMU_MOSI_COVAREP",
    "text": "CMU_MOSI_TimestampedWordVectors",
    "raw": "CMU_MOSI_TimestampedWords",
    "visual": "CMU_MOSI_Visual_OpenFace_2",
    "labels": "CMU_MOSI_Opinion_Labels",
}

MOSI_OPENSMILE_FACET_RAW = {
    "audio": MOSI_OPENSMILE_FEATS,
    "text": "CMU_MOSI_TimestampedWords",
    "raw": "CMU_MOSI_TimestampedWords",
    "visual": "CMU_MOSI_Visual_Facet_42",
    "labels": "CMU_MOSI_Opinion_Labels",
}

MOSI_OPENSMILE_FACET_GLOVE = {
    "audio": MOSI_OPENSMILE_FEATS,
    "text": "CMU_MOSI_TimestampedWordVectors",
    "raw": "CMU_MOSI_TimestampedWords",
    "visual": "CMU_MOSI_Visual_Facet_42",
    "labels": "CMU_MOSI_Opinion_Labels",
}

MOSI_OPENSMILE_OPENFACE_RAW = {
    "audio": MOSI_OPENSMILE_FEATS,
    "text": "CMU_MOSI_TimestampedWords",
    "raw": "CMU_MOSI_TimestampedWords",
    "visual": "CMU_MOSI_Visual_OpenFace_2",
    "labels": "CMU_MOSI_Opinion_Labels",
}

MOSI_OPENSMILE_OPENFACE_GLOVE = {
    "audio": MOSI_OPENSMILE_FEATS,
    "text": "CMU_MOSI_TimestampedWordVectors",
    "raw": "CMU_MOSI_TimestampedWords",
    "visual": "CMU_MOSI_Visual_OpenFace_2",
    "labels": "CMU_MOSI_Opinion_Labels",
}

# MOSEI

MOSEI_COVAREP_FACET_RAW = {
    "audio": "CMU_MOSEI_COVAREP",
    "text": "CMU_MOSEI_TimestampedWords",
    "raw": "CMU_MOSEI_TimestampedWords",
    "visual": "CMU_MOSEI_VisualFacet42",
    "labels": "CMU_MOSEI_Labels",
}

MOSEI_COVAREP_FACET_GLOVE = {
    "audio": "CMU_MOSEI_COVAREP",
    "text": "CMU_MOSEI_TimestampedWordVectors",
    "raw": "CMU_MOSEI_TimestampedWords",
    "visual": "CMU_MOSEI_VisualFacet42",
    "labels": "CMU_MOSEI_Labels",
}

MOSEI_COVAREP_OPENFACE_RAW = {
    "audio": "CMU_MOSEI_COVAREP",
    "text": "CMU_MOSEI_TimestampedWords",
    "raw": "CMU_MOSEI_TimestampedWords",
    "visual": "CMU_MOSEI_VisualOpenFace2",
    "labels": "CMU_MOSEI_Labels",
}

MOSEI_COVAREP_OPENFACE_GLOVE = {
    "audio": "CMU_MOSEI_COVAREP",
    "text": "CMU_MOSEI_TimestampedWordVectors",
    "raw": "CMU_MOSEI_TimestampedWords",
    "visual": "CMU_MOSEI_VisualOpenFace2",
    "labels": "CMU_MOSEI_Labels",
}

# POM

POM_COVAREP_FACET_RAW = {
    "audio": "POM_COVAREP",
    "text": "POM_TimestampedWords",
    "raw": "POM_TimestampedWords",
    "visual": "POM_Facet_42",
    "labels": "POM_Labels",
}

POM_COVAREP_FACET_GLOVE = {
    "audio": "POM_COVAREP",
    "text": "POM_TimestampedWordVectors",
    "raw": "POM_TimestampedWords",
    "visual": "POM_Facet_42",
    "labels": "POM_Labels",
}

POM_COVAREP_OPENFACE_RAW = {
    "audio": "POM_COVAREP",
    "text": "POM_TimestampedWords",
    "raw": "POM_TimestampedWords",
    "visual": "POM_OpenFace2",
    "labels": "POM_Labels",
}

POM_COVAREP_OPENFACE_GLOVE = {
    "audio": "POM_COVAREP",
    "text": "POM_TimestampedWordVectors",
    "raw": "POM_TimestampedWords",
    "visual": "POM_OpenFace2",
    "labels": "POM_Labels",
}
