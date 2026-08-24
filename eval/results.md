# MLSC Knowledge Assistant -- Evaluation Results

LLM provider used for this run: `ollama`
Total questions: 18

## Aggregate metrics

| Metric | Score |
|---|---|
| Context Precision (answerable questions only, n=14) | 0.85 |
| Context Recall (answerable questions only, n=14) | 0.93 |
| Answer Relevancy (all questions) | 0.99 |
| Faithfulness / Groundedness (all questions) | 0.99 |
| Answerable/Unanswerable Detection Accuracy | 1.00 |

## Per-category breakdown

| Category | n | Avg Precision | Avg Recall | Avg Relevancy | Avg Faithfulness | Detection Accuracy |
|---|---|---|---|---|---|---|
| ambiguous | 2 | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 |
| direct | 5 | 0.77 | 1.00 | 1.00 | 1.00 | 1.00 |
| multi_doc | 4 | 0.92 | 0.88 | 0.95 | 0.95 | 1.00 |
| reasoning | 3 | 1.00 | 0.83 | 1.00 | 1.00 | 1.00 |
| unanswerable | 4 | n/a | n/a | 1.00 | 1.00 | 1.00 |

## Per-question results

| ID | Category | Question | Expected Sources | Retrieved Sources | Precision | Recall | Relevancy | Faithfulness | Answerable OK |
|---|---|---|---|---|---|---|---|---|---|
| 1 | direct | What technical domains exist in MLSC? | domains.txt | domains.txt, leadership.txt | 0.50 | 1.00 | 1.00 | 1.00 | yes |
| 2 | direct | What is MLSC and what does it focus on? | about_mlsc.txt | membership.txt, code_of_conduct.txt, about_mlsc.txt | 0.33 | 1.00 | 1.00 | 1.00 | yes |
| 3 | direct | What stages do participants typically go through in an MLSC hackathon? | hackathons.txt | hackathons.txt | 1.00 | 1.00 | 1.00 | 1.00 | yes |
| 4 | direct | What kinds of behavior are not acceptable under the MLSC code of conduct? | code_of_conduct.txt | code_of_conduct.txt | 1.00 | 1.00 | 1.00 | 1.00 | yes |
| 5 | direct | What are domain leads responsible for? | leadership.txt | leadership.txt | 1.00 | 1.00 | 1.00 | 1.00 | yes |
| 6 | multi_doc | How does the domain lead structure relate to the technical domains in MLSC? | leadership.txt, domains.txt | leadership.txt, domains.txt | 1.00 | 1.00 | 1.00 | 1.00 | yes |
| 7 | multi_doc | If I want to build an AI-powered project for an MLSC hackathon, which domains might be involved and what stages would I go through? | hackathons.txt, domains.txt | hackathons.txt, domains.txt | 1.00 | 1.00 | 1.00 | 1.00 | yes |
| 8 | multi_doc | What is expected of members in terms of conduct, and how does someone move from being a member to a coordinator? | code_of_conduct.txt, membership.txt | membership.txt, code_of_conduct.txt, leadership.txt | 0.67 | 1.00 | 0.80 | 0.80 | yes |
| 9 | multi_doc | How are domain leads and coordinators related, and what does someone need to do to become a coordinator? | leadership.txt, membership.txt | leadership.txt | 1.00 | 0.50 | 1.00 | 1.00 | yes |
| 10 | reasoning | Based on what domain leads are responsible for, what kind of skills would a coordinator need to develop before being ready for a domain lead role? | leadership.txt | leadership.txt | 1.00 | 1.00 | 1.00 | 1.00 | yes |
| 11 | reasoning | Why might a hackathon project need contributions from more than one MLSC technical domain? | hackathons.txt, domains.txt | hackathons.txt | 1.00 | 0.50 | 1.00 | 1.00 | yes |
| 12 | reasoning | Why does the MLSC code of conduct encourage challenging ideas constructively rather than avoiding technical disagreement altogether? | code_of_conduct.txt | code_of_conduct.txt | 1.00 | 1.00 | 1.00 | 1.00 | yes |
| 13 | unanswerable | What is the capital of France? | - | domains.txt, leadership.txt, about_mlsc.txt | n/a | n/a | 1.00 | 1.00 | yes |
| 14 | unanswerable | Who is the current Technical Head of MLSC? | - | leadership.txt, code_of_conduct.txt, membership.txt | n/a | n/a | 1.00 | 1.00 | yes |
| 15 | unanswerable | Is there a registration fee for MLSC hackathons? | - | membership.txt, hackathons.txt, code_of_conduct.txt | n/a | n/a | 1.00 | 1.00 | yes |
| 16 | unanswerable | Does MLSC have chapters in countries outside India? | - | about_mlsc.txt, membership.txt, code_of_conduct.txt | n/a | n/a | 1.00 | 1.00 | yes |
| 17 | ambiguous | What are the various technology areas MLSC works in? | domains.txt | domains.txt, about_mlsc.txt, hackathons.txt | 0.33 | 1.00 | 1.00 | 1.00 | yes |
| 18 | ambiguous | How can a student join MLSC and get involved? | membership.txt | membership.txt | 1.00 | 1.00 | 1.00 | 1.00 | yes |