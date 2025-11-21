# Multilingual few-shot examples for labeling
# Use these in your LLM prompt as short examples to improve multilingual understanding.

### Example 1 (English)
Title: Simone Biles wins Olympic vault final
Tags: olympics, gymnastics, simone biles
Label: olympic
Reason: explicitly about the Olympic Games and an Olympic athlete.

### Example 2 (Spanish)
Title: Medalla de oro en natación - Highlights
Tags: olimpiadas, natación
Label: olympic
Reason: mentions olimpiadas and medal in swimming.

### Example 3 (Portuguese)
Title: Melhores momentos: futebol brasileiro
Tags: futebol, gols
Label: other_sport
Reason: sports-related but not referencing Olympics.

### Example 4 (Japanese)
Title: サッカー最新ハイライト
Tags: サッカー, ゴール
Label: other_sport
Reason: football highlights; not Olympic-specific.

### Example 5 (Non-sport)
Title: New music video by pop star
Tags: music, official video
Label: non_sport
Reason: entertainment, not sports-related.

# Guidance:
# - Output strictly in machine-friendly form:
#   Label: <one of olympic | other_sport | non_sport>
#   Reason: <one-sentence reason>
#
# - Aim for very short reasons so parsing is easy.
