![covinfo](/img/covinfo_logo.png)

## Introduction
We have developed COVInfo, a knowledge platform that verifies information from scientific peer-reviewed journals in hopes of combating misinformation circulating the Internet in light of COVID-19. COVInfo will be available on GitHub, so that other developers can access it and develop their own tools to verify misinformation. With COVInfo, developers can create tools such as misinformation verifiers and search engine integrators. Unlike other knowledge graphs, COVInfo will only verify data against published scientific journals. We will first start using PubMed and then scale to include other scientific journals.

Given the circumstances of the pandemic, we want to make COVInfo accessible to the public, and it will be open source. A team of developers will help us maintain our knowledge graph and we will continue to grow with the assistance of government institutions. We encourage you to download COVInfo and start exploring ways you can tackle misinformation related to COVID-19.

## Development

To develop COVInfo, we had to run a database with a collection of tweets and news articles. A team of us searched through various communication channels, and we evaluated them as either being “true”, “false”, or “neutral.” We used the text extracted from these sources to develop COVInfo’s artificial intelligence. 

| Link                                                                                                                            | Text                                                                                                                                                                                                                                                                                                                                                           | Classification | Topic                |
|---------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|----------------------|
| https://www.sciencealert.com/the-new-coronavirus-could-have-been-percolating-innocently-in-humans-for-years                     | The COVID-19 Virus May Have Been in Humans For Years, Study Suggest                                                                                                                                                                                                                                                                                            | False          | Bio-Engineered Virus |
| https://www.livemint.com/news/world/nobel-winning-scientist-claims-covid-19-virus-was-man-made-in-wuhan-lab-11587303649821.html | Nobel winning scientist claims Covid-19 virus was man-made in Wuhan lab                                                                                                                                                                                                                                                                                        | False          | Bio-Engineered Virus |
| https://www.sciencemag.org/news/2020/02/scientists-strongly-condemn-rumors-and-conspiracy-theories-about-origin-coronavirus     | Scientists ‘strongly condemn’ rumors and conspiracy theories about origin of coronavirus outbreak                                                                                                                                                                                                                                                              | True           | Bio-Engineered Virus |
| https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30418-9/fulltext                                             | Scientists from multiple countries have published and analysed genomes of the causative agent, severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), and they overwhelmingly conclude that this coronavirus originated in wildlife,                                                                                                                    | True           | Bio-Engineered Virus |
| https://www.bbc.com/news/health-53077893                                                                                        | Chief investigator Prof Peter Horby said it was "the only drug so far that has been shown to reduce mortality - and it reduces it significantly.”                                                                                                                                                                                                              | True           | Dexamethasone        |
| https://doctormurray.com/does-the-flu-shot-increase-covid-19-risk/                                                              | There is evidence that influenza vaccines specifically increase the risk of coronavirus infection. Here is why, a phenomenon known as virus interference. Yes, it appears that the flu shot protects against influenza and it appears some other types of viruses as well, but it comes at a price of actually increasing the risk for coronavirus infections. | False          | Flu vaccine          |

While searching through articles and tweets, we selected sources, which we could verify with strong evidence arguing against them or in support of them. We identified news articles as being “true, when they were backed with reliable sources, such as other peer reviewed journals. We identified the “false” articles by conducting general searches of current conspiracy theories, and then finding articles supporting them. We ensured that the conspiracy theories these articles supported had been dispelled by scientists. Sources were deemed “neutral” when they presented information that was once approved by scientists, but were then re-considered with more evidence -- for example, the BBC encouraged readers to take paracetamol before ibuprofen for mild coronavirus symptoms. This was once advised by the NHS, but later studies showed that ibuprofen did not worsen symptoms. Taking paracetamol first also does not pose any great threats to readers, so we regarded the information as “neutral.” (https://www.bbc.com/news/health-52894638)

Additionally, we classified articles based on common topics related to coronavirus, for example, Hydroxychloroquine, conspiracy theories about the virus’ origin, and Vitamin D’s impact on patients’ defenses.

## Use case: Fackt-checking

With a knowledge graph approach, we can extract entities and their relationships in a statement. Take for instance this tweet (https://twitter.com/olivierveran/status/1238776545398923264?s=20):

***“The taking of anti-inflammatories [ibuprofen, cortisone … ] could be a factor in aggravating the infection. In case of fever, take paracetamol. If you are already taking anti-inflammatory drugs, ask your doctor’s advice”***

From this claim, we can construct a graph in this manner:

![claim](/img/claim_graph.png)

However, a publication by Sridharan, G. K., Kotagiri, R., Chandiramani, et al. (2020) claim that actually the use of ibuprofen may be beneficial to treat COVID-19. The actual statement is the following:

***“However, available data from limited studies show administration of recombinant ACE2 improves lung damage caused by respiratory viruses, suggesting ibuprofen use may be beneficial in COVID-19 disease. At this time, there is no supporting evidence to discourage the use of ibuprofen.”***

From this statement we can construct the following graph:

![fact](/img/fact_graph.png)

Having these graphs, we can construct the following statements: ("Ibuprofen", "aggravates", "COVID-19") and ("Ibuprofen", "improves", "COVID-19").

Whit these, we can do a sentence-pairs classification to know is they contradict or not:

```python
from fact_checker.sentence_pair_classification import fact_check

claim = "Ibuprofen aggravates COVID-19"
fact = "Ibuprofen improves COVID-19"

prediction, check = fact_check(claim, fact)
print(check)
```
The out for this example must be "False", meaning that the claim was contradicted by the fact founded in the publication.


