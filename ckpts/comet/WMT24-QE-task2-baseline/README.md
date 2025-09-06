---
extra_gated_heading: Acknowledge license to accept the repository
extra_gated_button_content: Acknowledge license
language:
- multilingual
license: cc-by-nc-sa-4.0
library_name: transformers
---

Baseline for subtask 2 (word-level) from [WMT24 Quality Estimation shared task](https://www2.statmt.org/wmt24/qe-task.html). This baseline follows the approach described in [Unbabel-IST submission from 2022: CometKiwi](https://aclanthology.org/2022.wmt-1.60) (Rei et al., WMT 2022)

# License:

cc-by-nc-sa-4.0

# Usage (unbabel-comet)

Using this model requires installing an older fork of unbabel-comet:

```bash
git clone https://github.com/ricardorei/wmt22-comet-legacy
poetry install
```

Make sure you acknowledge its License and Log in into Hugging face hub before using:

```bash
huggingface-cli login
# or using an environment variable
huggingface-cli login --token $HUGGINGFACE_TOKEN
```

### Running the model:

```python
from comet import download_model, load_from_checkpoint

model_path = download_model("Unbabel/wmt22-cometkiwi-da")
model = load_from_checkpoint(model_path)
data = [
    {
        "src": "The output signal provides constant sync so the display never glitches.",
        "mt": "Das Ausgangssignal bietet eine konstante Synchronisation, so dass die Anzeige nie stört."
    },
    {
        "src": "Kroužek ilustrace je určen všem milovníkům umění ve věku od 10 do 15 let.",
        "mt": "Кільце ілюстрації призначене для всіх любителів мистецтва у віці від 10 до 15 років."
    },
    {
        "src": "Mandela then became South Africa's first black president after his African National Congress party won the 1994 election.",
        "mt": "その後、1994年の選挙でアフリカ国民会議派が勝利し、南アフリカ初の黒人大統領となった。"
    }
]
model_output = model.predict(data, batch_size=8, gpus=1)
print (model_output)
```

# Intended uses

Our model is intented to be used for **reference-free MT evaluation**. 

Given a source text and its translation, outputs a single score and word-level OK/BAD tags.

The model is a baseline for a shared task and it should only be used for research purposes.

# Languages Covered:

This model builds on top of InfoXLM which cover the following languages:

Afrikaans, Albanian, Amharic, Arabic, Armenian, Assamese, Azerbaijani, Basque, Belarusian, Bengali, Bengali Romanized, Bosnian, Breton, Bulgarian, Burmese, Burmese, Catalan, Chinese (Simplified), Chinese (Traditional), Croatian, Czech, Danish, Dutch, English, Esperanto, Estonian, Filipino, Finnish, French, Galician, Georgian, German, Greek, Gujarati, Hausa, Hebrew, Hindi, Hindi Romanized, Hungarian, Icelandic, Indonesian, Irish, Italian, Japanese, Javanese, Kannada, Kazakh, Khmer, Korean, Kurdish (Kurmanji), Kyrgyz, Lao, Latin, Latvian, Lithuanian, Macedonian, Malagasy, Malay, Malayalam, Marathi, Mongolian, Nepali, Norwegian, Oriya, Oromo, Pashto, Persian, Polish, Portuguese, Punjabi, Romanian, Russian, Sanskri, Scottish, Gaelic, Serbian, Sindhi, Sinhala, Slovak, Slovenian, Somali, Spanish, Sundanese, Swahili, Swedish, Tamil, Tamil Romanized, Telugu, Telugu Romanized, Thai, Turkish, Ukrainian, Urdu, Urdu Romanized, Uyghur, Uzbek, Vietnamese, Welsh, Western, Frisian, Xhosa, Yiddish.

Thus, results for language pairs containing uncovered languages are unreliable!