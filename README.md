# Knowledge Extraction with open-source LLMs

Congratulations to joining a tremendous learning experiment and thank you for choosing the E.ON Track! We're glad to invite you into the World of Digital Energy Solutions!

Take your seats and fasten the belts. We are starting... 3 -> 2 -> 1!

# Challenges for E.ON

Automation of text processing has been getting increased attention over the last months, especially in light of the rise of Generative models.

On the market, there are many competitors for commercial solutions.
E.ON partners with Microsoft Azure incorporating Azure OpenAI models into its platform (E.ON GPT) and a familily of custom solutions.
Additinally, E.ON consideres other commercial models from big players like Amazon and Google.

Current situation presents a set of challeges:
- commercial offerings are limited in deployable capacity, it is not always possible to get requested bandwith (tokens-per-minute) for acceptable User experience;
- evaluated solutions are expensive, and the costs rise with the number of users;
- it is difficult to adapt models, finetuning is restricted or even not possible;
- the is no way to host own specialized model based on a foreign base model.


## Question Answering as Reading Comprehension

For everybody new in the topic we recommend to read and understand the following book chapter:
https://web.stanford.edu/~jurafsky/slp3/14.pdf

See references at the bottom may help as detailed introduction to specific topics.

## Data Sheet

Valid questions:
```json
{
  "title": "Beyoncé",
  "paragraphs": [
    {
      "qas": [
        {
          "question": "When did Beyonce start becoming popular?",
          "id": "56be85543aeaaa14008c9063",
          "answers": [
            {
              "text": "in the late 1990s",
              "answer_start": 269
            }
          ],
          "is_impossible": false
        },
        {
          "question": "What areas did Beyonce compete in when she was growing up?",
          "id": "56be85543aeaaa14008c9065",
          "answers": [
            {
              "text": "singing and dancing",
              "answer_start": 207
            }
          ],
          "is_impossible": false
        }
      ],
      "context": "Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyoncé's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles \"Crazy in Love\" and \"Baby Boy\"."
    }
  ]
}
```

Artificial questions:
```json
{
  "qas": [
    {
        "plausible_answers": [
            {
                "text": "November 2006",
                "answer_start": 569
            }
        ],
        "question": "When could GameCube owners purchase Australian Princess?",
        "id": "5a8d7bf7df8bba001a0f9ab4",
        "answers": [],
        "is_impossible": true
    },
    {
        "plausible_answers": [
            {
                "text": "2005",
                "answer_start": 364
            }
        ],
        "question": "What year was the Legend of Zelda: Australian Princess originally planned for release?",
        "id": "5a8d7bf7df8bba001a0f9ab5",
        "answers": [],
        "is_impossible": true
    }
],
  "context": "The Legend of Zelda: Twilight Princess (Japanese: ゼルダの伝説 トワイライトプリンセス, Hepburn: Zeruda no Densetsu: Towairaito Purinsesu?) is an action-adventure
game developed and published by Nintendo for the GameCube and Wii home video game consoles. It is the thirteenth installment in the The Legend of Zelda series. Originally planned for
 release on the GameCube in November 2005, Twilight Princess was delayed by Nintendo to allow its developers to refine the game, add more content, and port it to the Wii. The Wii ver
sion was released alongside the console in North America in November 2006, and in Japan, Europe, and Australia the following month. The GameCube version was released worldwide in Dec
ember 2006.[b]"
}
```

Training/Finetuning
- German: `GermanQUaD/GermanQuAD_train.json`
- English: `SQuAD/train-v2.0.json`

Evaluation:
- German: `GermanQUaD/GermanQuAD_test.json`
- English: `SQuAD/dev-v2.0.json`

See original descriptions:
- SQuAD 1.0: https://arxiv.org/abs/1606.05250
- SQuAD 2.0: https://arxiv.org/abs/1806.03822
- GermanQuAD: https://arxiv.org/pdf/2104.12741.pdf

## Use Cases

The challenge is built around one main NLP task: Question Answering over unstructured textual Data.

Complexity levels:
- Level 0: Choose one open-source LLM, use it AS IS for Answer generation given questions from the dev part of SQuAD 1.0 and evaluate it using the provided evaluation script.
- Level 1: On SQuAD 1.0 (only valid questions) use the chosen LLM and augment the retrieval by means of model finetuning and/or RAG technique, calculate F1 for comparison.
- Level 3: On SQuAD 2.0 (valid and imaginary questions) employ results of the previous level with indication to imaginary answers.

Bonus levels:
- Level 0+: Compare several (3+) models in their raw performance.
- Level 0+: Look for a general purpose model with good performance for German and use it on GermanQUaD.
- level 1+: Compute not only the F1 metric, but also BLUE and an other appropriate metric (research question), compare the results.
- Level 1+: Adapt a German model for GermanQUaD data.

The participants are expected to provide a solution at one level at least starting with the Level 0 since the latter tasks are based on the former. It is up to participants to to wide or deep in the task definition.

A "solution" can be a running software solution or a kind of Proof-of-Concept run manually in form of a Notebook.

## Infrastructure

Participants are supposed to explore "smaller" models which are approachable on local computes (decent GPU for a gaming PC).

Otherwise open offering (Kaggle, Google Colab) can be used with the proposed data:
- https://www.kaggle.com/docs/efficient-gpu-usage
- https://research.google.com/colaboratory/faq.html#gpu-availability

## Evaluation

Use the standard script `SQuAD/evaluate-v2.0.py` to calculate the F1 score. Additionally calculate/implement your own appropriate metrics.

See `SQuAD/dev-evaluate-v2.0-in1` for an example input for the evaluation script.

Use the dev part of the corpus for the evaluation.

The leaderboard: https://rajpurkar.github.io/SQuAD-explorer/

## References

Corpus:
- https://rajpurkar.github.io/SQuAD-explorer/
- https://www.deepset.ai/germanquad

NLP Courses:
- https://course.spacy.io/en/
- https://huggingface.co/learn/nlp-course/

Tutorials:
- https://huggingface.co/learn/nlp-course/chapter7/7

Frameworks:
- https://ollama.com/
- https://docs.chainlit.io/

Fine-tuning:
- https://www.philschmid.de/fsdp-qlora-llama3
- https://github.com/teticio/llama-squad

Models:
- https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
- https://huggingface.co/models
- https://chat.lmsys.org/?leaderboard

## Our expectations

We expect the participants to stay in touch with the mentors during the whole term. Please do continuously spend time on the tasks and do not hope to tackle everything in the last minute!
We are open to your questions and hope to provide as much support as possible to you given your motivation and dedication to the challenge topic.
And you'll experience that even bigger elephants can get swallowed in small pieces by chewing them carefully over longer time :)
Have fun and happy hacking!
