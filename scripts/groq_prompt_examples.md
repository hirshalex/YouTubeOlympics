You are labeling YouTube videos.
Look at the examples, then follow the same JSON format exactly.

Examples:
Title: Simone Biles wins Olympic vault final
Tags: olympics, gymnastics
-> {"label":"olympic","reason":"about Olympic Games"}

Title: Melhores momentos: futebol brasileiro
Tags: futebol, gols
-> {"label":"other_sport","reason":"sports video, not Olympic"}

Title: New music video by pop star
Tags: music, official video
-> {"label":"non_sport","reason":"music video, not sports"}

Now label this video. Respond only with one JSON object and nothing else:

Title: <<title>>
Tags: <<tags>>
